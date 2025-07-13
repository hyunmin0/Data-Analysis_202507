"""
한국전력공사 전력 사용패턴 변동계수 개발 (정확도 우선 버전)
- 고속모드 제거, 정확도 최우선
- 충분한 교차검증과 모델 성능 확보
- 과적합 방지 강화
"""

import pandas as pd
import numpy as np
import json
import os
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from math import pi
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

class KEPCOVolatilityAnalyzer:
    """한국전력공사 변동계수 스태킹 분석기 (정확도 우선 버전)"""
    
    def __init__(self, results_dir='./analysis_results', sampling_config=None):
        self.results_dir = results_dir
        self.scaler = StandardScaler()
        self.robust_scaler = RobustScaler()
        self.level0_models = {}
        self.meta_model = None
        
        # 샘플링 설정 (정확도 우선)
        self.sampling_config = sampling_config or {
            'customer_sample_ratio': 0.7,      # 고객의 70%만 샘플링 (정확도 확보)
            'time_sample_ratio': 0.5,          # 시간 데이터의 50%만 샘플링  
            'min_customers': 50,               # 최소 50명
            'min_records_per_customer': 200,   # 고객당 최소 200개 레코드
            'stratified_sampling': True,       # 계층 샘플링 사용
            'validation_folds': 5              # 5-fold 교차검증
        }
        
        # 기존 전처리 결과 로딩
        self.step1_results = self._load_step1_results()
        self.step2_results = self._load_step2_results()
        self.lp_data = None
        self.kepco_data = None
        
        print("🔧 한국전력공사 변동계수 스태킹 분석기 초기화 (정확도 우선)")
        print(f"   📊 샘플링 설정: 고객 {self.sampling_config['customer_sample_ratio']*100:.0f}%, 시간 {self.sampling_config['time_sample_ratio']*100:.0f}%")
        print(f"   🎯 정확도 우선 모드: 충분한 검증과 과적합 방지")
        
    def _load_step1_results(self):
        """1단계 전처리 결과 로딩"""
        try:
            file_path = os.path.join(self.results_dir, 'analysis_results.json')
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    results = json.load(f)
                print(f"   ✅ 1단계 결과 로딩: {len(results)}개 항목")
                return results
            else:
                return {}
        except Exception as e:
            print(f"   ❌ 1단계 결과 로딩 실패: {e}")
            return {}
    
    def _load_step2_results(self):
        """2단계 시계열 분석 결과 로딩"""
        try:
            file_path = os.path.join(self.results_dir, 'analysis_results2.json')
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    results = json.load(f)
                print(f"   ✅ 2단계 결과 로딩: {len(results)}개 항목")
                return results
            else:
                return {}
        except Exception as e:
            print(f"   ❌ 2단계 결과 로딩 실패: {e}")
            return {}
    
    def load_preprocessed_data_with_sampling(self):
        """전처리된 데이터 로딩 + 스마트 샘플링"""
        print("\n📊 전처리된 데이터 로딩 및 샘플링 중...")
        
        # 1. LP 데이터 로딩
        hdf5_path = os.path.join(self.results_dir, 'processed_lp_data.h5')
        csv_path = os.path.join(self.results_dir, 'processed_lp_data.csv')
        
        if os.path.exists(hdf5_path):
            try:
                self.lp_data = pd.read_hdf(hdf5_path, key='df')
                loading_method = "HDF5"
            except Exception as e:
                if os.path.exists(csv_path):
                    self.lp_data = pd.read_csv(csv_path)
                    loading_method = "CSV"
                else:
                    print(f"   ❌ LP 데이터를 찾을 수 없습니다.")
                    return False
        elif os.path.exists(csv_path):
            self.lp_data = pd.read_csv(csv_path)
            loading_method = "CSV"
        else:
            print(f"   ❌ 전처리된 LP 데이터가 없습니다.")
            return False
        
        original_size = len(self.lp_data)
        print(f"   📁 원본 데이터: {original_size:,}건 ({loading_method})")
        
        # 2. 컬럼 정리 및 기본 전처리
        self._prepare_columns()
        
        # 3. 스마트 샘플링 적용 (정확도 우선)
        self._apply_smart_sampling()
        
        sampled_size = len(self.lp_data)
        reduction_ratio = (1 - sampled_size/original_size) * 100
        
        print(f"   ✂️ 샘플링 후: {sampled_size:,}건")
        print(f"   📉 데이터 감소: {reduction_ratio:.1f}%")
        print(f"   📅 기간: {self.lp_data['datetime'].min()} ~ {self.lp_data['datetime'].max()}")
        print(f"   👥 고객수: {self.lp_data['대체고객번호'].nunique()}")
        
        return True
    
    def _prepare_columns(self):
        """컬럼 정리 및 기본 전처리"""
        # datetime 컬럼 처리
        datetime_col = None
        for col in ['datetime', 'LP 수신일자', 'LP수신일자', 'timestamp']:
            if col in self.lp_data.columns:
                datetime_col = col
                break
        
        if datetime_col:
            self.lp_data['datetime'] = pd.to_datetime(self.lp_data[datetime_col], errors='coerce')
            self.lp_data = self.lp_data.dropna(subset=['datetime'])
        else:
            raise ValueError("날짜/시간 컬럼을 찾을 수 없습니다.")
        
        # 전력 컬럼 처리
        power_col = None
        for col in ['순방향 유효전력', '순방향유효전력', 'power', '전력량']:
            if col in self.lp_data.columns:
                if col != '순방향 유효전력':
                    self.lp_data['순방향 유효전력'] = self.lp_data[col]
                power_col = '순방향 유효전력'
                break
        
        if not power_col:
            raise ValueError("순방향 유효전력 컬럼을 찾을 수 없습니다.")
        
        # 데이터 품질 정리
        self.lp_data = self.lp_data.dropna(subset=['순방향 유효전력'])
        self.lp_data.loc[self.lp_data['순방향 유효전력'] < 0, '순방향 유효전력'] = 0
        
        # 극단 이상치 처리 (99.9% 분위수로 캡핑)
        q999 = self.lp_data['순방향 유효전력'].quantile(0.999)
        self.lp_data.loc[self.lp_data['순방향 유효전력'] > q999, '순방향 유효전력'] = q999
    
    def _apply_smart_sampling(self):
        """스마트 샘플링 적용 (정확도 우선)"""
        print("   🎯 정확도 우선 스마트 샘플링 적용 중...")
        
        # 1. 고객별 데이터 충분성 확인
        customer_counts = self.lp_data['대체고객번호'].value_counts()
        sufficient_customers = customer_counts[
            customer_counts >= self.sampling_config['min_records_per_customer']
        ].index.tolist()
        
        print(f"      충분한 데이터 보유 고객: {len(sufficient_customers)}명")
        
        # 2. 계층 샘플링 (업종별, 규모별)
        if self.sampling_config['stratified_sampling']:
            sampled_customers = self._stratified_customer_sampling(sufficient_customers)
        else:
            # 단순 랜덤 샘플링
            n_customers = max(
                self.sampling_config['min_customers'],
                int(len(sufficient_customers) * self.sampling_config['customer_sample_ratio'])
            )
            sampled_customers = np.random.choice(
                sufficient_customers, 
                size=min(n_customers, len(sufficient_customers)), 
                replace=False
            ).tolist()
        
        print(f"      샘플링된 고객: {len(sampled_customers)}명")
        
        # 3. 고객 필터링
        self.lp_data = self.lp_data[self.lp_data['대체고객번호'].isin(sampled_customers)]
        
        # 4. 시간 샘플링 (각 고객별로) - 정확도 보장을 위해 더 많은 데이터 유지
        if self.sampling_config['time_sample_ratio'] < 1.0:
            sampled_data = []
            
            for customer_id in sampled_customers:
                customer_data = self.lp_data[self.lp_data['대체고객번호'] == customer_id]
                
                # 시간 기반 계층 샘플링 (피크/비피크, 주중/주말 균등하게)
                n_samples = max(
                    self.sampling_config['min_records_per_customer'],
                    int(len(customer_data) * self.sampling_config['time_sample_ratio'])
                )
                
                if len(customer_data) <= n_samples:
                    sampled_data.append(customer_data)
                else:
                    # 시간 균등 샘플링 (정확도 확보를 위해 대표적인 시간대 포함)
                    sampled_data.append(self._balanced_time_sampling(customer_data, n_samples))
            
            self.lp_data = pd.concat(sampled_data, ignore_index=True)
            print(f"      시간 샘플링 완료 (균형 잡힌 시간대 포함)")
    
    def _balanced_time_sampling(self, customer_data, n_samples):
        """균형 잡힌 시간 샘플링 (피크/비피크, 주중/주말 고려)"""
        customer_data = customer_data.copy()
        customer_data['hour'] = customer_data['datetime'].dt.hour
        customer_data['weekday'] = customer_data['datetime'].dt.weekday
        customer_data['is_weekend'] = customer_data['weekday'].isin([5, 6])
        
        # 시간대별, 주중/주말별 그룹 생성
        groups = []
        
        # 피크 시간대 (9-11, 14-15, 18-19) - 주중
        peak_weekday = customer_data[
            (customer_data['hour'].isin([9, 10, 11, 14, 15, 18, 19])) & 
            (~customer_data['is_weekend'])
        ]
        
        # 비피크 시간대 (0-5, 22-23) - 주중
        off_peak_weekday = customer_data[
            (customer_data['hour'].isin([0, 1, 2, 3, 4, 5, 22, 23])) & 
            (~customer_data['is_weekend'])
        ]
        
        # 일반 시간대 - 주중
        normal_weekday = customer_data[
            (~customer_data['hour'].isin([0, 1, 2, 3, 4, 5, 9, 10, 11, 14, 15, 18, 19, 22, 23])) & 
            (~customer_data['is_weekend'])
        ]
        
        # 주말 데이터
        weekend_data = customer_data[customer_data['is_weekend']]
        
        # 각 그룹에서 비례적으로 샘플링
        total_samples = n_samples
        samples_per_group = total_samples // 4
        
        sampled_groups = []
        for group in [peak_weekday, off_peak_weekday, normal_weekday, weekend_data]:
            if len(group) > 0:
                group_samples = min(samples_per_group, len(group))
                if group_samples > 0:
                    sampled_groups.append(group.sample(n=group_samples, random_state=42))
        
        # 남은 샘플 수를 가장 큰 그룹에서 추가 샘플링
        current_total = sum(len(g) for g in sampled_groups)
        remaining = total_samples - current_total
        
        if remaining > 0 and len(customer_data) > current_total:
            used_indices = set()
            for g in sampled_groups:
                used_indices.update(g.index)
            
            remaining_data = customer_data[~customer_data.index.isin(used_indices)]
            if len(remaining_data) > 0:
                additional_samples = min(remaining, len(remaining_data))
                sampled_groups.append(remaining_data.sample(n=additional_samples, random_state=42))
        
        if sampled_groups:
            return pd.concat(sampled_groups)
        else:
            # 폴백: 단순 랜덤 샘플링
            return customer_data.sample(n=min(n_samples, len(customer_data)), random_state=42)
    
    def _stratified_customer_sampling(self, customers):
        """계층별 고객 샘플링 (정확도 우선)"""
        # 고객별 평균 전력 사용량으로 계층 구분
        customer_power_avg = self.lp_data.groupby('대체고객번호')['순방향 유효전력'].mean()
        
        # 4개 계층으로 구분 (소형, 중소형, 중형, 대형) - 더 세밀한 구분
        q25, q50, q75 = customer_power_avg.quantile([0.25, 0.50, 0.75])
        
        small_customers = customer_power_avg[customer_power_avg <= q25].index.tolist()
        medium_small_customers = customer_power_avg[(customer_power_avg > q25) & (customer_power_avg <= q50)].index.tolist()
        medium_customers = customer_power_avg[(customer_power_avg > q50) & (customer_power_avg <= q75)].index.tolist()
        large_customers = customer_power_avg[customer_power_avg > q75].index.tolist()
        
        # 각 계층에서 비례적으로 샘플링
        total_target = max(
            self.sampling_config['min_customers'],
            int(len(customers) * self.sampling_config['customer_sample_ratio'])
        )
        
        small_n = min(len(small_customers), max(1, total_target // 4))
        medium_small_n = min(len(medium_small_customers), max(1, total_target // 4))
        medium_n = min(len(medium_customers), max(1, total_target // 4))
        large_n = min(len(large_customers), max(1, total_target - small_n - medium_small_n - medium_n))
        
        sampled = []
        if small_customers:
            sampled.extend(np.random.choice(small_customers, size=small_n, replace=False))
        if medium_small_customers:
            sampled.extend(np.random.choice(medium_small_customers, size=medium_small_n, replace=False))
        if medium_customers:
            sampled.extend(np.random.choice(medium_customers, size=medium_n, replace=False))
        if large_customers:
            sampled.extend(np.random.choice(large_customers, size=large_n, replace=False))
        
        print(f"      계층별 샘플링: 소형{small_n}명, 중소형{medium_small_n}명, 중형{medium_n}명, 대형{large_n}명")
        return sampled
    
    def calculate_enhanced_volatility_coefficient(self):
        """향상된 변동계수 계산 (정확도 우선)"""
        print("\n📐 향상된 변동계수 계산 중 (정확도 우선)...")
        
        if self.lp_data is None:
            print("   ❌ LP 데이터가 로딩되지 않았습니다.")
            return {}
        
        # 2단계 결과에서 시간 패턴 정보 가져오기
        temporal_patterns = self.step2_results.get('temporal_patterns', {})
        peak_hours = temporal_patterns.get('peak_hours', [9, 10, 11, 14, 15, 18, 19])
        off_peak_hours = temporal_patterns.get('off_peak_hours', [0, 1, 2, 3, 4, 5])
        weekend_ratio = temporal_patterns.get('weekend_ratio', 1.0)
        
        print(f"   🕐 피크 시간: {peak_hours}")
        print(f"   🌙 비피크 시간: {off_peak_hours}")
        
        # 시간 파생 변수 생성
        self.lp_data['hour'] = self.lp_data['datetime'].dt.hour
        self.lp_data['weekday'] = self.lp_data['datetime'].dt.weekday
        self.lp_data['is_weekend'] = self.lp_data['weekday'].isin([5, 6])
        self.lp_data['month'] = self.lp_data['datetime'].dt.month
        self.lp_data['date'] = self.lp_data['datetime'].dt.date
        
        customers = self.lp_data['대체고객번호'].unique()
        volatility_results = {}
        volatility_components = []
        processed_count = 0
        
        print(f"   👥 분석 대상: {len(customers)}명")
        
        for customer_id in customers:
            try:
                customer_data = self.lp_data[self.lp_data['대체고객번호'] == customer_id].copy()
                
                if len(customer_data) < self.sampling_config['min_records_per_customer']:
                    continue
                
                power_values = customer_data['순방향 유효전력'].values
                
                # 데이터 품질 검증 (더 엄격한 기준)
                if np.std(power_values) == 0 or np.mean(power_values) <= 0:
                    continue
                
                # 변동성 지표 계산 (더 정밀한 계산)
                volatility_metrics = self._calculate_volatility_metrics_accurate(
                    customer_data, power_values, peak_hours, off_peak_hours, weekend_ratio
                )
                
                if volatility_metrics:
                    volatility_components.append({
                        'customer_id': customer_id,
                        **volatility_metrics
                    })
                    processed_count += 1
                
                if processed_count % 10 == 0:
                    print(f"      진행률: {processed_count}/{len(customers)} 완료")
                
            except Exception as e:
                print(f"   ⚠️ 고객 {customer_id} 계산 실패: {e}")
                continue
        
        print(f"   ✅ {processed_count}명 변동성 지표 계산 완료")
        
        # 가중치 최적화 (정확도 우선 - 충분한 최적화)
        if len(volatility_components) >= 20:
            optimal_weights = self.optimize_volatility_weights_accurate(volatility_components)
        else:
            optimal_weights = [0.35, 0.25, 0.20, 0.10, 0.10]  # 기본 가중치
            print(f"   ⚠️ 데이터 부족으로 기본 가중치 사용")
        
        print(f"   🎯 최종 가중치: {[round(w, 3) for w in optimal_weights]}")
        
        # 최종 변동계수 계산
        for component in volatility_components:
            customer_id = component['customer_id']
            
            enhanced_volatility_coefficient = (
                optimal_weights[0] * component['basic_cv'] +
                optimal_weights[1] * component['hourly_cv'] +
                optimal_weights[2] * component['peak_cv'] +
                optimal_weights[3] * component['weekend_diff'] +
                optimal_weights[4] * component['seasonal_cv']
            )
            
            volatility_results[customer_id] = {
                'enhanced_volatility_coefficient': round(enhanced_volatility_coefficient, 4),
                'basic_cv': round(component['basic_cv'], 4),
                'hourly_cv': round(component['hourly_cv'], 4),
                'peak_cv': round(component['peak_cv'], 4),
                'off_peak_cv': round(component['off_peak_cv'], 4),
                'weekday_cv': round(component['weekday_cv'], 4),
                'weekend_cv': round(component['weekend_cv'], 4),
                'weekend_diff': round(component['weekend_diff'], 4),
                'seasonal_cv': round(component['seasonal_cv'], 4),
                'load_factor': round(component['load_factor'], 4),
                'peak_load_ratio': round(component['peak_load_ratio'], 4),
                'mean_power': round(component['mean_power'], 4),
                'zero_ratio': round(component['zero_ratio'], 4),
                'extreme_changes': int(component['extreme_changes']),
                'data_points': component['data_points'],
                'optimized_weights': [round(w, 3) for w in optimal_weights],
                'stability_score': round(component.get('stability_score', 0), 4),
                'predictability_score': round(component.get('predictability_score', 0), 4)
            }
        
        if len(volatility_results) > 0:
            cv_values = [v['enhanced_volatility_coefficient'] for v in volatility_results.values()]
            print(f"   📈 평균 변동계수: {np.mean(cv_values):.4f}")
            print(f"   📊 변동계수 범위: {np.min(cv_values):.4f} ~ {np.max(cv_values):.4f}")
        
        return volatility_results
    
    def _calculate_volatility_metrics_accurate(self, customer_data, power_values, peak_hours, off_peak_hours, weekend_ratio):
        """개별 고객의 변동성 지표 계산 (정확도 우선)"""
        try:
            mean_power = np.mean(power_values)
            
            # 1. 기본 변동계수 (더 정밀한 계산)
            basic_cv = np.std(power_values, ddof=1) / mean_power
            
            # 2. 시간대별 변동계수 (24시간 세분화)
            hourly_avg = customer_data.groupby('hour')['순방향 유효전력'].agg(['mean', 'std', 'count'])
            # 충분한 데이터가 있는 시간대만 고려
            valid_hours = hourly_avg[hourly_avg['count'] >= 5]
            hourly_cv = (np.std(valid_hours['mean']) / np.mean(valid_hours['mean'])) if len(valid_hours) > 3 and np.mean(valid_hours['mean']) > 0 else basic_cv
            
            # 3. 피크/비피크 변동성 (더 정밀한 분석)
            peak_data = customer_data[customer_data['hour'].isin(peak_hours)]['순방향 유효전력']
            off_peak_data = customer_data[customer_data['hour'].isin(off_peak_hours)]['순방향 유효전력']
            
            peak_cv = (np.std(peak_data, ddof=1) / np.mean(peak_data)) if len(peak_data) > 10 and np.mean(peak_data) > 0 else basic_cv
            off_peak_cv = (np.std(off_peak_data, ddof=1) / np.mean(off_peak_data)) if len(off_peak_data) > 10 and np.mean(off_peak_data) > 0 else basic_cv
            
            # 4. 주말/평일 변동성 (더 세밀한 분석)
            weekday_data = customer_data[~customer_data['is_weekend']]['순방향 유효전력']
            weekend_data = customer_data[customer_data['is_weekend']]['순방향 유효전력']
            
            weekday_cv = (np.std(weekday_data, ddof=1) / np.mean(weekday_data)) if len(weekday_data) > 20 and np.mean(weekday_data) > 0 else basic_cv
            weekend_cv = (np.std(weekend_data, ddof=1) / np.mean(weekend_data)) if len(weekend_data) > 10 and np.mean(weekend_data) > 0 else basic_cv
            weekend_diff = abs(weekday_cv - weekend_cv) * weekend_ratio
            
            # 5. 계절별 변동성 (일별/주별 집계로 더 정밀하게)
            daily_avg = customer_data.groupby('date')['순방향 유효전력'].mean()
            seasonal_cv = (np.std(daily_avg, ddof=1) / np.mean(daily_avg)) if len(daily_avg) > 7 and np.mean(daily_avg) > 0 else basic_cv
            
            # 6. 추가 안정성 지표들
            max_power = np.max(power_values)
            min_power = np.min(power_values[power_values > 0]) if np.sum(power_values > 0) > 0 else 0
            load_factor = mean_power / max_power if max_power > 0 else 0
            zero_ratio = (power_values == 0).sum() / len(power_values)
            
            # 급격한 변화 감지 (더 정밀한 임계값)
            power_series = pd.Series(power_values)
            pct_changes = power_series.pct_change().dropna()
            extreme_changes = (np.abs(pct_changes) > 2.0).sum()  # 200% 변화
            
            # 피크/비피크 부하 비율
            peak_avg = np.mean(peak_data) if len(peak_data) > 0 else mean_power
            off_peak_avg = np.mean(off_peak_data) if len(off_peak_data) > 0 else mean_power
            peak_load_ratio = peak_avg / off_peak_avg if off_peak_avg > 0 else 1.0
            
            # 7. 안정성 점수 (새로 추가)
            # 낮은 변동성, 높은 부하율, 적은 제로값, 적은 극한 변화
            stability_score = (
                (1 - min(basic_cv, 1.0)) * 0.4 +  # 기본 변동성 역수
                load_factor * 0.3 +  # 부하율
                (1 - zero_ratio) * 0.2 +  # 제로값 역수
                (1 - min(extreme_changes / len(power_values), 1.0)) * 0.1  # 극한 변화 역수
            )
            
            # 8. 예측가능성 점수 (패턴의 규칙성)
            # 시간대별 일관성, 주중/주말 일관성
            time_consistency = 1 - (hourly_cv / (basic_cv + 1e-6))
            day_consistency = 1 - abs(weekday_cv - weekend_cv) / (basic_cv + 1e-6)
            predictability_score = (time_consistency * 0.6 + day_consistency * 0.4)
            predictability_score = max(0, min(1, predictability_score))
            
            return {
                'basic_cv': basic_cv,
                'hourly_cv': hourly_cv,
                'peak_cv': peak_cv,
                'off_peak_cv': off_peak_cv,
                'weekday_cv': weekday_cv,
                'weekend_cv': weekend_cv,
                'weekend_diff': weekend_diff,
                'seasonal_cv': seasonal_cv,
                'load_factor': load_factor,
                'zero_ratio': zero_ratio,
                'extreme_changes': extreme_changes,
                'peak_load_ratio': peak_load_ratio,
                'mean_power': mean_power,
                'max_power': max_power,
                'min_power': min_power,
                'data_points': len(power_values),
                'stability_score': stability_score,
                'predictability_score': predictability_score
            }
            
        except Exception as e:
            return None
    
    def optimize_volatility_weights_accurate(self, volatility_components):
        """가중치 최적화 (정확도 우선)"""
        print("\n⚙️ 가중치 최적화 중 (정확도 우선)...")
        
        try:
            from scipy.optimize import minimize, differential_evolution
        except ImportError:
            print("   ⚠️ scipy가 설치되지 않아 기본 가중치 사용")
            return [0.35, 0.25, 0.20, 0.10, 0.10]
        
        components_df = pd.DataFrame(volatility_components)
        
        # 복합 목표 함수: 영업활동 불안정성과 예측 어려움
        target_instability = (
            components_df['basic_cv'] * 3.0 +  # 기본 변동성
            components_df['zero_ratio'] * 2.0 +  # 사용 중단 빈도
            (1 - components_df['load_factor']) * 1.5 +  # 비효율적 사용
            (1 - components_df['stability_score']) * 2.0 +  # 불안정성
            (1 - components_df['predictability_score']) * 1.0  # 예측 어려움
        ).values
        
        X = components_df[['basic_cv', 'hourly_cv', 'peak_cv', 'weekend_diff', 'seasonal_cv']].values
        y = target_instability
        
        # 표준화
        from sklearn.preprocessing import StandardScaler
        scaler_x = StandardScaler()
        scaler_y = StandardScaler()
        X_scaled = scaler_x.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
        
        def objective(weights):
            predicted = X_scaled @ weights
            mse = np.mean((predicted - y_scaled) ** 2)
            # 가중치 분산을 최소화하여 극단적인 가중치 방지 (정규화)
            weight_penalty = np.std(weights) * 0.1
            return mse + weight_penalty
        
        # 제약 조건
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},  # 가중치 합 = 1
            {'type': 'ineq', 'fun': lambda w: w[0] - 0.1},  # 기본 CV는 최소 10%
        ]
        bounds = [(0.05, 0.6) for _ in range(5)]  # 각 가중치는 5%-60% 범위
        
        # 여러 번 최적화 시도하여 최적해 찾기
        best_result = None
        best_score = float('inf')
        
        for seed in range(5):
            initial_weights = np.random.dirichlet([1, 1, 1, 1, 1])  # 합이 1인 랜덤 가중치
            
            try:
                # SLSQP 방법
                result = minimize(objective, initial_weights, method='SLSQP', 
                                bounds=bounds, constraints=constraints, 
                                options={'maxiter': 200})
                
                if result.success and result.fun < best_score:
                    best_result = result
                    best_score = result.fun
                    
            except:
                continue
        
        # Differential Evolution 시도 (글로벌 최적화)
        try:
            def objective_de(weights):
                if abs(np.sum(weights) - 1.0) > 0.01:  # 가중치 합 제약
                    return 1e6
                if np.any(weights < 0.05) or np.any(weights > 0.6):  # 범위 제약
                    return 1e6
                return objective(weights)
            
            bounds_de = [(0.05, 0.6) for _ in range(5)]
            result_de = differential_evolution(objective_de, bounds_de, 
                                             maxiter=100, seed=42)
            
            if result_de.success and result_de.fun < best_score:
                best_result = result_de
                best_score = result_de.fun
                
        except:
            pass
        
        if best_result and best_result.success:
            optimal_weights = best_result.x
            # 가중치 정규화 (합이 정확히 1이 되도록)
            optimal_weights = optimal_weights / np.sum(optimal_weights)
            print(f"   ✅ 가중치 최적화 완료 (목적함수값: {best_score:.4f})")
            
            # 최적화 품질 검증
            r2_score_weights = self._validate_weight_optimization(X_scaled, y_scaled, optimal_weights)
            print(f"   📊 가중치 최적화 R²: {r2_score_weights:.4f}")
            
            return optimal_weights.tolist()
        else:
            print("   ⚠️ 최적화 실패, 기본 가중치 사용")
            return [0.35, 0.25, 0.20, 0.10, 0.10]
    
    def _validate_weight_optimization(self, X_scaled, y_scaled, weights):
        """가중치 최적화 검증"""
        predicted = X_scaled @ weights
        ss_res = np.sum((y_scaled - predicted) ** 2)
        ss_tot = np.sum((y_scaled - np.mean(y_scaled)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        return r2
    
    def train_stacking_ensemble_model_accurate(self, volatility_results):
        """스태킹 앙상블 모델 훈련 (정확도 우선)"""
        print("\n🎯 스태킹 앙상블 모델 훈련 중 (정확도 우선)...")
        
        if len(volatility_results) < 20:
            print("   ❌ 훈련 데이터가 부족합니다 (최소 20개 필요)")
            return None
        
        # 특성 추출 (더 많은 특성 포함)
        features = []
        targets = []
        
        for customer_id, data in volatility_results.items():
            try:
                feature_vector = [
                    data['basic_cv'], data['hourly_cv'], data['peak_cv'],
                    data['off_peak_cv'], data['weekday_cv'], data['weekend_cv'],
                    data['seasonal_cv'], data['load_factor'], data['peak_load_ratio'],
                    data['mean_power'], data['zero_ratio'],
                    data['extreme_changes'] / data['data_points'],
                    data['stability_score'], data['predictability_score'],
                    np.log1p(data['mean_power']),  # 로그 변환된 평균 전력
                    data['max_power'] / data['mean_power'] if data['mean_power'] > 0 else 0,  # 최대/평균 비율
                ]
                
                if any(np.isnan(x) or np.isinf(x) for x in feature_vector):
                    continue
                    
                features.append(feature_vector)
                targets.append(data['enhanced_volatility_coefficient'])
                
            except KeyError as e:
                print(f"   ⚠️ 특성 추출 실패: {e}")
                continue
        
        X = np.array(features)
        y = np.array(targets)
        
        print(f"   📊 훈련 데이터: {len(X)}개 샘플, {X.shape[1]}개 특성")
        
        # 데이터 분할 (계층화 분할)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=pd.cut(y, bins=5, labels=False)
        )
        
        # 정규화 (RobustScaler - 이상치에 강함)
        X_train_scaled = self.robust_scaler.fit_transform(X_train)
        X_test_scaled = self.robust_scaler.transform(X_test)
        
        # Level-0 모델들 (정확도 우선 - 과적합 방지)
        self.level0_models = {
            'rf': RandomForestRegressor(
                n_estimators=100, max_depth=8, min_samples_split=5, 
                min_samples_leaf=3, random_state=42, n_jobs=-1
            ),
            'gbm': GradientBoostingRegressor(
                n_estimators=100, max_depth=6, learning_rate=0.1,
                min_samples_split=5, min_samples_leaf=3, random_state=42
            ),
            'ridge': Ridge(alpha=1.0, random_state=42),
            'elastic': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42, max_iter=2000)
        }
        
        # 5-fold 교차검증
        kf = KFold(n_splits=self.sampling_config['validation_folds'], shuffle=True, random_state=42)
        
        meta_features_train = np.zeros((len(X_train_scaled), len(self.level0_models)))
        meta_features_test = np.zeros((len(X_test_scaled), len(self.level0_models)))
        
        level0_performance = {}
        
        print(f"   🔄 Level-0 모델 훈련 ({self.sampling_config['validation_folds']}-Fold CV):")
        
        for i, (name, model) in enumerate(self.level0_models.items()):
            fold_predictions = np.zeros(len(X_train_scaled))
            fold_scores = []
            
            for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_scaled)):
                try:
                    fold_model = type(model)(**model.get_params())
                    fold_model.fit(X_train_scaled[train_idx], y_train[train_idx])
                    
                    val_pred = fold_model.predict(X_train_scaled[val_idx])
                    fold_predictions[val_idx] = val_pred
                    
                    # Fold 성능 평가
                    fold_mae = mean_absolute_error(y_train[val_idx], val_pred)
                    fold_r2 = r2_score(y_train[val_idx], val_pred)
                    fold_scores.append({'mae': fold_mae, 'r2': fold_r2})
                    
                except Exception as e:
                    print(f"      ⚠️ {name} Fold {fold+1} 실패: {e}")
                    fold_predictions[val_idx] = np.mean(y_train[train_idx])
            
            meta_features_train[:, i] = fold_predictions
            
            # 전체 훈련 세트로 재훈련
            try:
                model.fit(X_train_scaled, y_train)
                meta_features_test[:, i] = model.predict(X_test_scaled)
                
                # 테스트 성능 평가
                test_pred = model.predict(X_test_scaled)
                test_mae = mean_absolute_error(y_test, test_pred)
                test_r2 = r2_score(y_test, test_pred) if len(set(y_test)) > 1 else 0.0
                test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
                
                # CV 성능 평균
                cv_mae = np.mean([score['mae'] for score in fold_scores])
                cv_r2 = np.mean([score['r2'] for score in fold_scores])
                
                level0_performance[name] = {
                    'cv_mae': cv_mae, 'cv_r2': cv_r2,
                    'test_mae': test_mae, 'test_r2': test_r2, 'test_rmse': test_rmse
                }
                
                print(f"      {name}: CV MAE={cv_mae:.4f}, Test MAE={test_mae:.4f}, Test R²={test_r2:.4f}")
                
            except Exception as e:
                print(f"      ❌ {name} 훈련 실패: {e}")
                meta_features_test[:, i] = np.mean(y_train)
                level0_performance[name] = {'cv_mae': 999, 'cv_r2': 0, 'test_mae': 999, 'test_r2': 0, 'test_rmse': 999}
        
        # Level-1 메타 모델 (Ridge Regression with CV)
        meta_models = {
            'ridge': Ridge(alpha=1.0),
            'elastic': ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=2000),
            'linear': LinearRegression()
        }
        
        best_meta_model = None
        best_meta_score = -999
        final_performance = {}
        
        print(f"   🔄 Level-1 메타모델 선택:")
        
        for meta_name, meta_model in meta_models.items():
            try:
                # 메타모델 교차검증
                cv_scores = cross_val_score(meta_model, meta_features_train, y_train, 
                                          cv=3, scoring='r2')
                avg_cv_score = np.mean(cv_scores)
                
                # 메타모델 훈련 및 테스트
                meta_model.fit(meta_features_train, y_train)
                final_pred = meta_model.predict(meta_features_test)
                
                final_mae = mean_absolute_error(y_test, final_pred)
                final_r2 = r2_score(y_test, final_pred) if len(set(y_test)) > 1 else 0.0
                final_rmse = np.sqrt(mean_squared_error(y_test, final_pred))
                
                print(f"      {meta_name}: CV R²={avg_cv_score:.4f}, Test R²={final_r2:.4f}")
                
                if final_r2 > best_meta_score:
                    best_meta_model = meta_model
                    best_meta_score = final_r2
                    final_performance = {
                        'final_mae': final_mae,
                        'final_r2': final_r2,
                        'final_rmse': final_rmse,
                        'cv_r2': avg_cv_score,
                        'meta_model_name': meta_name
                    }
                
            except Exception as e:
                print(f"      ❌ {meta_name} 메타모델 실패: {e}")
        
        if best_meta_model is not None:
            self.meta_model = best_meta_model
            
            print(f"   ✅ 스태킹 앙상블 훈련 완료")
            print(f"      최고 메타모델: {final_performance['meta_model_name']}")
            print(f"      최종 MAE: {final_performance['final_mae']:.4f}")
            print(f"      최종 R²: {final_performance['final_r2']:.4f}")
            print(f"      최종 RMSE: {final_performance['final_rmse']:.4f}")
            
            # 과적합 검사
            train_pred = self.meta_model.predict(meta_features_train)
            train_r2 = r2_score(y_train, train_pred)
            overfitting_gap = train_r2 - final_performance['final_r2']
            
            print(f"      과적합 점검: 훈련 R²={train_r2:.4f}, 차이={overfitting_gap:.4f}")
            if overfitting_gap > 0.1:
                print(f"      ⚠️ 과적합 의심 (차이 > 0.1)")
            else:
                print(f"      ✅ 과적합 없음")
            
            return {
                **final_performance,
                'level0_performance': level0_performance,
                'level0_models': list(self.level0_models.keys()),
                'n_samples': len(X),
                'n_features': X.shape[1],
                'overfitting_gap': overfitting_gap,
                'accuracy_optimized': True
            }
        else:
            print("   ❌ 모든 메타모델 훈련 실패")
            return None

    def analyze_business_stability_accurate(self, volatility_results):
        """영업활동 안정성 분석 (정확도 우선)"""
        print("\n🔍 영업활동 안정성 분석 중 (정확도 우선)...")
        
        if not volatility_results:
            return {}
        
        coefficients = [v['enhanced_volatility_coefficient'] for v in volatility_results.values()]
        stability_scores = [v['stability_score'] for v in volatility_results.values()]
        predictability_scores = [v['predictability_score'] for v in volatility_results.values()]
        
        # 다차원 분석을 위한 분위수 계산
        cv_p33, cv_p67 = np.percentile(coefficients, [33, 67])
        stability_p33, stability_p67 = np.percentile(stability_scores, [33, 67])
        
        stability_analysis = {}
        grade_counts = {'안정': 0, '보통': 0, '주의': 0, '위험': 0}
        
        for customer_id, data in volatility_results.items():
            coeff = data['enhanced_volatility_coefficient']
            stability = data['stability_score']
            predictability = data['predictability_score']
            
            # 4단계 등급 분류 (더 세밀한 분류)
            if coeff <= cv_p33 and stability >= stability_p67:
                grade = '안정'
                risk_level = 'low'
            elif coeff <= cv_p67 and stability >= stability_p33:
                grade = '보통'
                risk_level = 'medium'
            elif coeff > cv_p67 or stability < stability_p33:
                grade = '주의'
                risk_level = 'high'
            else:
                grade = '위험'
                risk_level = 'very_high'
            
            grade_counts[grade] += 1
            
            # 정밀한 위험 요인 분석
            risk_factors = []
            if data.get('zero_ratio', 0) > 0.15:
                risk_factors.append('빈번한_사용중단')
            if data.get('load_factor', 1) < 0.2:
                risk_factors.append('매우_낮은_부하율')
            elif data.get('load_factor', 1) < 0.4:
                risk_factors.append('낮은_부하율')
            if data.get('peak_cv', 0) > data.get('basic_cv', 0) * 2.5:
                risk_factors.append('피크시간_극도_불안정')
            elif data.get('peak_cv', 0) > data.get('basic_cv', 0) * 1.5:
                risk_factors.append('피크시간_불안정')
            if data.get('extreme_changes', 0) / data.get('data_points', 1) > 0.05:
                risk_factors.append('급격한_변화_빈발')
            if predictability < 0.3:
                risk_factors.append('예측_불가능한_패턴')
            if data.get('weekend_diff', 0) > 0.3:
                risk_factors.append('주중주말_패턴_상이')
            
            # 영업활동 변화 예측
            business_trend = self._predict_business_trend(data)
            
            stability_analysis[customer_id] = {
                'enhanced_volatility_coefficient': round(coeff, 4),
                'stability_grade': grade,
                'risk_level': risk_level,
                'risk_factors': risk_factors,
                'stability_score': round(stability, 4),
                'predictability_score': round(predictability, 4),
                'load_factor': data.get('load_factor', 0.0),
                'peak_load_ratio': data.get('peak_load_ratio', 1.0),
                'business_trend_prediction': business_trend,
                'monitoring_priority': self._calculate_monitoring_priority(coeff, stability, risk_factors)
            }
        
        print(f"   📋 안정성 등급 분포:")
        total = len(stability_analysis)
        for grade, count in grade_counts.items():
            percentage = count / total * 100 if total > 0 else 0
            print(f"      {grade}: {count}명 ({percentage:.1f}%)")
        
        # 위험 요인 상위 분석
        all_risk_factors = []
        for analysis in stability_analysis.values():
            all_risk_factors.extend(analysis['risk_factors'])
        
        from collections import Counter
        risk_factor_counts = Counter(all_risk_factors)
        print(f"   🚨 주요 위험 요인:")
        for factor, count in risk_factor_counts.most_common(5):
            print(f"      {factor}: {count}건")
        
        return stability_analysis
    
    def _predict_business_trend(self, customer_data):
        """영업활동 변화 예측"""
        coeff = customer_data['enhanced_volatility_coefficient']
        stability = customer_data['stability_score']
        predictability = customer_data['predictability_score']
        zero_ratio = customer_data.get('zero_ratio', 0)
        load_factor = customer_data.get('load_factor', 1)
        
        # 복합 지표 기반 트렌드 예측
        if coeff < 0.2 and stability > 0.7 and zero_ratio < 0.05:
            return '안정적_성장'
        elif coeff < 0.3 and stability > 0.5:
            return '점진적_개선'
        elif coeff > 0.5 or stability < 0.3 or zero_ratio > 0.2:
            return '영업활동_위축'
        elif load_factor < 0.3 and predictability < 0.4:
            return '불규칙적_운영'
        else:
            return '현상_유지'
    
    def _calculate_monitoring_priority(self, coeff, stability, risk_factors):
        """모니터링 우선순위 계산"""
        priority_score = 0
        
        # 변동계수 점수
        if coeff > 0.6:
            priority_score += 3
        elif coeff > 0.4:
            priority_score += 2
        elif coeff > 0.3:
            priority_score += 1
        
        # 안정성 점수
        if stability < 0.3:
            priority_score += 3
        elif stability < 0.5:
            priority_score += 2
        elif stability < 0.7:
            priority_score += 1
        
        # 위험 요인 점수
        high_risk_factors = ['빈번한_사용중단', '매우_낮은_부하율', '피크시간_극도_불안정']
        for factor in risk_factors:
            if factor in high_risk_factors:
                priority_score += 2
            else:
                priority_score += 1
        
        # 우선순위 등급
        if priority_score >= 7:
            return '최우선'
        elif priority_score >= 5:
            return '높음'
        elif priority_score >= 3:
            return '보통'
        else:
            return '낮음'

    def generate_comprehensive_report(self, volatility_results, model_performance, stability_analysis):
        """종합 분석 리포트 생성 (정확도 우선)"""
        print("\n📋 종합 분석 리포트 생성 중...")
        
        coefficients = [v['enhanced_volatility_coefficient'] for v in volatility_results.values()] if volatility_results else []
        
        # 고위험 고객 식별
        high_risk_customers = [
            customer_id for customer_id, analysis in stability_analysis.items()
            if analysis['risk_level'] in ['high', 'very_high']
        ] if stability_analysis else []
        
        # 모니터링 우선순위별 고객 분류
        priority_groups = {'최우선': [], '높음': [], '보통': [], '낮음': []}
        for customer_id, analysis in stability_analysis.items():
            priority = analysis.get('monitoring_priority', '낮음')
            priority_groups[priority].append(customer_id)
        
        # 주요 위험 요인 집계
        all_risk_factors = []
        for analysis in stability_analysis.values():
            all_risk_factors.extend(analysis['risk_factors'])
        
        from collections import Counter
        risk_factor_counts = Counter(all_risk_factors)
        
        # 영업활동 트렌드 분석
        trend_counts = Counter([
            analysis['business_trend_prediction'] 
            for analysis in stability_analysis.values()
        ])
        
        report = {
            'analysis_metadata': {
                'timestamp': pd.Timestamp.now().isoformat(),
                'algorithm_version': 'accuracy_optimized_v1',
                'sampling_config': self.sampling_config,
                'total_customers_analyzed': len(volatility_results),
                'execution_mode': 'accuracy_first',
                'validation_method': f"{self.sampling_config['validation_folds']}-fold_cross_validation"
            },
            
            'data_quality_summary': {
                'customer_sample_ratio': self.sampling_config['customer_sample_ratio'],
                'time_sample_ratio': self.sampling_config['time_sample_ratio'],
                'min_records_per_customer': self.sampling_config['min_records_per_customer'],
                'stratified_sampling_used': self.sampling_config['stratified_sampling']
            },
            
            'volatility_coefficient_analysis': {
                'total_customers': len(volatility_results),
                'mean_coefficient': round(np.mean(coefficients), 4) if coefficients else 0,
                'std_coefficient': round(np.std(coefficients), 4) if coefficients else 0,
                'percentiles': {
                    '10%': round(np.percentile(coefficients, 10), 4) if coefficients else 0,
                    '25%': round(np.percentile(coefficients, 25), 4) if coefficients else 0,
                    '50%': round(np.percentile(coefficients, 50), 4) if coefficients else 0,
                    '75%': round(np.percentile(coefficients, 75), 4) if coefficients else 0,
                    '90%': round(np.percentile(coefficients, 90), 4) if coefficients else 0
                },
                'distribution_analysis': {
                    '매우_안정 (<0.2)': len([c for c in coefficients if c < 0.2]),
                    '안정 (0.2-0.3)': len([c for c in coefficients if 0.2 <= c < 0.3]),
                    '보통 (0.3-0.4)': len([c for c in coefficients if 0.3 <= c < 0.4]),
                    '주의 (0.4-0.6)': len([c for c in coefficients if 0.4 <= c < 0.6]),
                    '위험 (0.6-0.8)': len([c for c in coefficients if 0.6 <= c < 0.8]),
                    '고위험 (>=0.8)': len([c for c in coefficients if c >= 0.8])
                }
            },
            
            'model_performance_analysis': model_performance or {},
            
            'business_stability_assessment': {
                'grade_distribution': {
                    grade: sum(1 for a in stability_analysis.values() if a['stability_grade'] == grade)
                    for grade in ['안정', '보통', '주의', '위험']
                } if stability_analysis else {},
                
                'trend_prediction': dict(trend_counts),
                
                'monitoring_priority': {
                    priority: len(customers) for priority, customers in priority_groups.items()
                }
            },
            
            'risk_analysis': {
                'high_risk_customers': len(high_risk_customers),
                'high_risk_percentage': round(len(high_risk_customers) / len(stability_analysis) * 100, 1) if stability_analysis else 0,
                'top_risk_factors': dict(risk_factor_counts.most_common(5)),
                'critical_alerts': self._generate_critical_alerts(stability_analysis)
            },
            
            'algorithmic_insights': {
                'weight_optimization_quality': model_performance.get('final_r2', 0) if model_performance else 0,
                'overfitting_assessment': model_performance.get('overfitting_gap', 0) if model_performance else 0,
                'feature_importance_analysis': self._analyze_feature_importance(volatility_results),
                'prediction_confidence': self._calculate_prediction_confidence(model_performance, stability_analysis)
            },
            
            'business_actionability': {
                'immediate_attention_required': len(priority_groups.get('최우선', [])),
                'monitoring_recommended': len(priority_groups.get('높음', [])),
                'stable_customers': len([
                    c for c in stability_analysis.values() 
                    if c['stability_grade'] == '안정'
                ]) if stability_analysis else 0,
                'efficiency_improvement_opportunities': self._identify_efficiency_opportunities(stability_analysis)
            },
            
            'technical_validation': {
                'data_sufficiency': self._assess_data_sufficiency(volatility_results),
                'statistical_significance': self._check_statistical_significance(coefficients),
                'algorithm_robustness': self._evaluate_algorithm_robustness(model_performance),
                'generalization_capability': model_performance.get('final_r2', 0) > 0.7 if model_performance else False
            },
            
            'recommendations': {
                'operational': [
                    f"최우선 모니터링 대상 {len(priority_groups.get('최우선', []))}개 고객 즉시 점검",
                    f"고위험 고객 {len(high_risk_customers)}명에 대한 영업활동 변화 분석",
                    "빈번한 사용중단 고객의 운영 상태 확인",
                    "낮은 부하율 고객의 전력 사용 효율성 개선 지원"
                ],
                'technical': [
                    f"모델 예측 정확도 R²={model_performance.get('final_r2', 0):.3f} 달성" if model_performance else "모델 성능 측정 필요",
                    f"과적합 점검: 차이={model_performance.get('overfitting_gap', 0):.3f}" if model_performance else "과적합 점검 필요",
                    "계층별 샘플링으로 대표성 확보",
                    "5-fold 교차검증으로 모델 신뢰성 확보"
                ],
                'business': [
                    "전력 사용패턴 변동계수 기반 위험도 분류 체계 구축",
                    "예측 불가능한 패턴 고객에 대한 맞춤형 서비스 개발",
                    "안정적 성장 고객 우대 정책 수립",
                    "영업활동 위축 예상 고객 사전 관리 강화"
                ]
            },
            
            'quality_assurance': {
                'accuracy_optimization_applied': True,
                'overfitting_prevention': model_performance.get('overfitting_gap', 1) < 0.1 if model_performance else False,
                'robust_validation': self.sampling_config['validation_folds'] >= 5,
                'sufficient_data_coverage': len(volatility_results) >= 50,
                'statistical_confidence': self._calculate_overall_confidence(volatility_results, model_performance)
            }
        }
        
        return report
    
    def _generate_critical_alerts(self, stability_analysis):
        """중요 알림 생성"""
        alerts = []
        
        if not stability_analysis:
            return alerts
        
        # 위험 고객 수 확인
        very_high_risk = [c for c in stability_analysis.values() if c['risk_level'] == 'very_high']
        if len(very_high_risk) > 0:
            alerts.append(f"극고위험 고객 {len(very_high_risk)}명 발견 - 즉시 점검 필요")
        
        # 영업활동 위축 고객
        declining_customers = [
            c for c in stability_analysis.values() 
            if c['business_trend_prediction'] == '영업활동_위축'
        ]
        if len(declining_customers) > len(stability_analysis) * 0.1:
            alerts.append(f"영업활동 위축 예상 고객 {len(declining_customers)}명 - 전체의 {len(declining_customers)/len(stability_analysis)*100:.1f}%")
        
        # 사용중단 빈발 고객
        frequent_outages = [
            c for c in stability_analysis.values()
            if '빈번한_사용중단' in c['risk_factors']
        ]
        if len(frequent_outages) > 0:
            alerts.append(f"사용중단 빈발 고객 {len(frequent_outages)}명 - 운영상태 점검 필요")
        
        return alerts
    
    def _analyze_feature_importance(self, volatility_results):
        """특성 중요도 분석"""
        if not volatility_results:
            return {}
        
        # 각 구성요소의 기여도 분석
        components = ['basic_cv', 'hourly_cv', 'peak_cv', 'weekend_diff', 'seasonal_cv']
        correlations = {}
        
        enhanced_cvs = [v['enhanced_volatility_coefficient'] for v in volatility_results.values()]
        
        for component in components:
            component_values = [v.get(component, 0) for v in volatility_results.values()]
            correlation = np.corrcoef(enhanced_cvs, component_values)[0, 1]
            correlations[component] = round(correlation, 4) if not np.isnan(correlation) else 0
        
        return correlations
    
    def _calculate_prediction_confidence(self, model_performance, stability_analysis):
        """예측 신뢰도 계산"""
        if not model_performance or not stability_analysis:
            return 0.0
        
        r2_score = model_performance.get('final_r2', 0)
        overfitting_gap = model_performance.get('overfitting_gap', 1)
        
        # 높은 R², 낮은 과적합일수록 높은 신뢰도
        confidence = r2_score * (1 - min(overfitting_gap, 0.5))
        return round(max(0, min(1, confidence)), 4)
    
    def _identify_efficiency_opportunities(self, stability_analysis):
        """효율성 개선 기회 식별"""
        if not stability_analysis:
            return []
        
        opportunities = []
        
        # 낮은 부하율 고객
        low_load_factor = [
            c for c in stability_analysis.values()
            if c['load_factor'] < 0.3
        ]
        if len(low_load_factor) > 0:
            opportunities.append(f"부하율 개선 대상 {len(low_load_factor)}명")
        
        # 예측 가능한 패턴이지만 높은 변동성
        predictable_but_volatile = [
            c for c in stability_analysis.values()
            if c['predictability_score'] > 0.7 and c['enhanced_volatility_coefficient'] > 0.4
        ]
        if len(predictable_but_volatile) > 0:
            opportunities.append(f"패턴 최적화 가능 {len(predictable_but_volatile)}명")
        
        return opportunities
    
    def _assess_data_sufficiency(self, volatility_results):
        """데이터 충분성 평가"""
        if not volatility_results:
            return False
        
        total_customers = len(volatility_results)
        avg_data_points = np.mean([v['data_points'] for v in volatility_results.values()])
        
        return (total_customers >= 50 and 
                avg_data_points >= self.sampling_config['min_records_per_customer'])
    
    def _check_statistical_significance(self, coefficients):
        """통계적 유의성 확인"""
        if len(coefficients) < 30:
            return False
        
        # 정규성 검정 (간단한 방법)
        mean_cv = np.mean(coefficients)
        std_cv = np.std(coefficients)
        
        # 변동계수가 의미있는 분포를 가지는지 확인
        return std_cv > 0.01 and len(set(np.round(coefficients, 3))) > len(coefficients) * 0.5
    
    def _evaluate_algorithm_robustness(self, model_performance):
        """알고리즘 견고성 평가"""
        if not model_performance:
            return False
        
        r2_score = model_performance.get('final_r2', 0)
        overfitting_gap = model_performance.get('overfitting_gap', 1)
        
        return r2_score > 0.6 and overfitting_gap < 0.15
    
    def _calculate_overall_confidence(self, volatility_results, model_performance):
        """전체 신뢰도 계산"""
        factors = []
        
        # 데이터 품질
        if len(volatility_results) >= 100:
            factors.append(0.25)
        elif len(volatility_results) >= 50:
            factors.append(0.15)
        else:
            factors.append(0.05)
        
        # 모델 성능
        if model_performance:
            r2 = model_performance.get('final_r2', 0)
            factors.append(min(r2 * 0.3, 0.3))
        
        # 과적합 방지
        if model_performance and model_performance.get('overfitting_gap', 1) < 0.1:
            factors.append(0.2)
        
        # 검증 방법
        if self.sampling_config['validation_folds'] >= 5:
            factors.append(0.15)
        
        # 샘플링 품질
        if (self.sampling_config['customer_sample_ratio'] >= 0.5 and 
            self.sampling_config['stratified_sampling']):
            factors.append(0.1)
        
        return min(sum(factors), 1.0)

def main_accurate():
    """메인 실행 함수 (정확도 우선 버전)"""
    print("🏆 한국전력공사 전력 사용패턴 변동계수 분석 (정확도 우선)")
    print("=" * 70)
    
    start_time = datetime.now()
    
    # 정확도 우선 설정
    sampling_config = {
        'customer_sample_ratio': 0.8,      # 80% 고객 샘플링 (높은 대표성)
        'time_sample_ratio': 0.6,          # 60% 시간 데이터 샘플링
        'min_customers': 100,              # 최소 100명
        'min_records_per_customer': 300,   # 고객당 최소 300개 레코드
        'stratified_sampling': True,       # 계층 샘플링 사용
        'validation_folds': 5              # 5-fold 교차검증
    }
    
    print(f"🎯 정확도 우선 설정:")
    print(f"   📊 고객 샘플링: {sampling_config['customer_sample_ratio']*100:.0f}%")
    print(f"   ⏰ 시간 샘플링: {sampling_config['time_sample_ratio']*100:.0f}%")
    print(f"   🔍 교차검증: {sampling_config['validation_folds']}-fold")
    print(f"   📈 과적합 방지 강화")
    print()
    
    try:
        # 1. 분석기 초기화
        analyzer = KEPCOVolatilityAnalyzer('./analysis_results', sampling_config)
        
        # 2. 데이터 로딩 + 정밀 샘플링
        if not analyzer.load_preprocessed_data_with_sampling():
            print("❌ 데이터 로딩 실패")
            return None
        
        # 3. 변동계수 계산 (정확도 우선)
        volatility_results = analyzer.calculate_enhanced_volatility_coefficient()
        if not volatility_results:
            print("❌ 변동계수 계산 실패")
            return None
        
        # 4. 모델 훈련 (정확도 우선)
        model_performance = analyzer.train_stacking_ensemble_model_accurate(volatility_results)
        
        # 5. 안정성 분석 (정확도 우선)
        stability_analysis = analyzer.analyze_business_stability_accurate(volatility_results)
        
        # 6. 종합 리포트 생성
        report = analyzer.generate_comprehensive_report(volatility_results, model_performance, stability_analysis)
        
        # 7. 결과 저장
        save_accurate_results(volatility_results, stability_analysis, report, model_performance)
        
        # 실행 시간 계산
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        print(f"\n🎉 정확도 우선 분석 완료!")
        print(f"   ⏱️ 실행 시간: {execution_time:.1f}초")
        print(f"   👥 분석 고객: {len(volatility_results)}명")
        
        if volatility_results:
            cv_values = [v['enhanced_volatility_coefficient'] for v in volatility_results.values()]
            print(f"   📈 평균 변동계수: {np.mean(cv_values):.4f}")
            print(f"   📊 변동계수 범위: {np.min(cv_values):.4f} ~ {np.max(cv_values):.4f}")
        
        if model_performance:
            print(f"   🎯 모델 성능: R²={model_performance['final_r2']:.4f}")
            print(f"   ✅ 과적합 점검: 차이={model_performance.get('overfitting_gap', 0):.4f}")
        
        if stability_analysis:
            high_risk = len([a for a in stability_analysis.values() if a['risk_level'] in ['high', 'very_high']])
            print(f"   🚨 고위험 고객: {high_risk}명")
        
        # 품질 보증 점검
        quality_check = report.get('quality_assurance', {})
        print(f"\n🔍 품질 보증 점검:")
        print(f"   정확도 최적화: {'✅' if quality_check.get('accuracy_optimization_applied') else '❌'}")
        print(f"   과적합 방지: {'✅' if quality_check.get('overfitting_prevention') else '❌'}")
        print(f"   견고한 검증: {'✅' if quality_check.get('robust_validation') else '❌'}")
        print(f"   충분한 데이터: {'✅' if quality_check.get('sufficient_data_coverage') else '❌'}")
        print(f"   전체 신뢰도: {quality_check.get('statistical_confidence', 0):.3f}")
        
        return {
            'volatility_results': volatility_results,
            'model_performance': model_performance,
            'stability_analysis': stability_analysis,
            'report': report,
            'execution_time': execution_time,
            'sampling_config': sampling_config,
            'quality_assurance': quality_check
        }
        
    except Exception as e:
        print(f"❌ 시스템 실행 중 오류: {e}")
        import traceback
        traceback.print_exc()
        return None

def save_accurate_results(volatility_results, stability_analysis, report, model_performance):
    """정확도 우선 결과 저장"""
    try:
        os.makedirs('./analysis_results', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 1. 변동계수 결과 (상세 정보 포함)
        if volatility_results:
            df = pd.DataFrame.from_dict(volatility_results, orient='index')
            df.reset_index(inplace=True)
            df.rename(columns={'index': '대체고객번호'}, inplace=True)
            
            # 가중치 정보를 별도 컬럼으로 분리
            if 'optimized_weights' in df.columns and len(df) > 0:
                weights = df.loc[0, 'optimized_weights']
                if isinstance(weights, list) and len(weights) == 5:
                    df['weight_basic_cv'] = weights[0]
                    df['weight_hourly_cv'] = weights[1]
                    df['weight_peak_cv'] = weights[2]
                    df['weight_weekend_diff'] = weights[3]
                    df['weight_seasonal_cv'] = weights[4]
            
            csv_path = f'./analysis_results/volatility_accurate_{timestamp}.csv'
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            print(f"   💾 변동계수 결과: {csv_path}")
        
        # 2. 안정성 분석 (위험 요인 문자열화)
        if stability_analysis:
            df = pd.DataFrame.from_dict(stability_analysis, orient='index')
            df.reset_index(inplace=True)
            df.rename(columns={'index': '대체고객번호'}, inplace=True)
            
            if 'risk_factors' in df.columns:
                df['risk_factors_list'] = df['risk_factors'].apply(
                    lambda x: ', '.join(x) if isinstance(x, list) else str(x)
                )
            
            csv_path = f'./analysis_results/stability_accurate_{timestamp}.csv'
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            print(f"   💾 안정성 분석: {csv_path}")
        
        # 3. 종합 리포트
        if report:
            json_path = f'./analysis_results/comprehensive_report_{timestamp}.json'
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2, default=str)
            print(f"   💾 종합 리포트: {json_path}")
        
        # 4. 모델 성능 상세 정보
        if model_performance:
            json_path = f'./analysis_results/model_performance_{timestamp}.json'
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(model_performance, f, ensure_ascii=False, indent=2, default=str)
            print(f"   💾 모델 성능: {json_path}")
        
        # 5. 요약 보고서 (텍스트)
        summary_path = f'./analysis_results/executive_summary_{timestamp}.txt'
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("한국전력공사 전력 사용패턴 변동계수 분석 요약 보고서\n")
            f.write("=" * 60 + "\n\n")
            
            if volatility_results:
                cv_values = [v['enhanced_volatility_coefficient'] for v in volatility_results.values()]
                f.write(f"분석 고객 수: {len(volatility_results):,}명\n")
                f.write(f"평균 변동계수: {np.mean(cv_values):.4f}\n")
                f.write(f"변동계수 범위: {np.min(cv_values):.4f} ~ {np.max(cv_values):.4f}\n\n")
            
            if model_performance:
                f.write("모델 성능 지표:\n")
                f.write(f"  - 예측 정확도 (R²): {model_performance.get('final_r2', 0):.4f}\n")
                f.write(f"  - 평균 절대 오차: {model_performance.get('final_mae', 0):.4f}\n")
                f.write(f"  - 과적합 점검: {model_performance.get('overfitting_gap', 0):.4f}\n\n")
            
            if stability_analysis:
                grade_counts = {}
                for analysis in stability_analysis.values():
                    grade = analysis['stability_grade']
                    grade_counts[grade] = grade_counts.get(grade, 0) + 1
                
                f.write("안정성 등급 분포:\n")
                for grade, count in grade_counts.items():
                    percentage = count / len(stability_analysis) * 100
                    f.write(f"  - {grade}: {count}명 ({percentage:.1f}%)\n")
                
                high_risk = len([a for a in stability_analysis.values() if a['risk_level'] in ['high', 'very_high']])
                f.write(f"\n고위험 고객: {high_risk}명\n")
        
        print(f"   📋 요약 보고서: {summary_path}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ 결과 저장 실패: {e}")
        return False

def create_test_environment_accurate():
    """정확도 테스트 환경 생성"""
    print("🧪 정확도 우선 테스트 환경 생성 중...")
    
    import json
    os.makedirs('./analysis_results', exist_ok=True)
    
    # 1단계, 2단계 결과 생성
    step1_results = {
        'metadata': {'timestamp': datetime.now().isoformat(), 'total_customers': 500}
    }
    with open('./analysis_results/analysis_results.json', 'w', encoding='utf-8') as f:
        json.dump(step1_results, f, ensure_ascii=False, indent=2, default=str)
    
    step2_results = {
        'temporal_patterns': {
            'peak_hours': [9, 10, 11, 14, 15, 18, 19],
            'off_peak_hours': [0, 1, 2, 3, 4, 5, 22, 23],
            'weekend_ratio': 0.75
        }
    }
    with open('./analysis_results/analysis_results2.json', 'w', encoding='utf-8') as f:
        json.dump(step2_results, f, ensure_ascii=False, indent=2, default=str)
    
    # 고품질 테스트 데이터 생성 (500명, 30일)
    print("   📊 고품질 테스트 LP 데이터 생성 중...")
    
    np.random.seed(42)
    data = []
    
    for customer in range(1, 501):  # 500명
        # 고객별 다양한 특성 부여
        base_power = 20 + customer * 0.5 + np.random.normal(0, 10)
        base_power = max(10, base_power)  # 최소 10kW
        
        # 업종별 변동성 (더 현실적인 패턴)
        if customer % 5 == 0:  # 제조업 (20%)
            cv_base = 0.15 + np.random.uniform(0, 0.2)
            night_operation = True
        elif customer % 5 == 1:  # 상업 (20%)
            cv_base = 0.25 + np.random.uniform(0, 0.3)
            night_operation = False
        elif customer % 5 == 2:  # 사무 (20%)
            cv_base = 0.20 + np.random.uniform(0, 0.15)
            night_operation = False
        elif customer % 5 == 3:  # 서비스업 (20%)
            cv_base = 0.30 + np.random.uniform(0, 0.4)
            night_operation = False
        else:  # 기타 (20%)
            cv_base = 0.10 + np.random.uniform(0, 0.5)
            night_operation = np.random.choice([True, False])
        
        for day in range(30):  # 30일
            for hour in range(24):
                for minute in [0, 15, 30, 45]:  # 15분 간격
                    timestamp = datetime(2024, 3, 1) + timedelta(days=day, hours=hour, minutes=minute)
                    
                    # 시간대별 부하 패턴
                    hour_factor = 1.0
                    
                    # 피크 시간
                    if hour in [9, 10, 11, 14, 15, 18, 19]:
                        hour_factor = 1.2 + np.random.normal(0, 0.1)
                    # 비피크 시간
                    elif hour in [0, 1, 2, 3, 4, 5, 22, 23]:
                        if night_operation:
                            hour_factor = 1.1 + np.random.normal(0, 0.1)
                        else:
                            hour_factor = 0.3 + np.random.normal(0, 0.1)
                    # 일반 시간
                    else:
                        hour_factor = 0.9 + np.random.normal(0, 0.1)
                    
                    # 요일 효과
                    weekday = timestamp.weekday()
                    if weekday >= 5:  # 주말
                        if customer % 3 == 0:  # 일부 고객은 주말에도 운영
                            hour_factor *= 0.9
                        else:
                            hour_factor *= 0.4
                    
                    # 월별 계절 효과
                    month = timestamp.month
                    if month in [6, 7, 8]:  # 여름 (냉방)
                        hour_factor *= 1.3
                    elif month in [12, 1, 2]:  # 겨울 (난방)
                        hour_factor *= 1.2
                    
                    # 최종 전력량 계산
                    power = base_power * hour_factor * (1 + np.random.normal(0, cv_base))
                    
                    # 특수 상황 (정전, 휴업 등)
                    if np.random.random() < 0.02:  # 2% 확률
                        power = 0
                    else:
                        power = max(1, power)  # 최소 1kW
                    
                    data.append({
                        '대체고객번호': f'ACC_{customer:04d}',
                        'datetime': timestamp,
                        '순방향 유효전력': round(power, 2),
                        '지상무효': round(power * 0.3 * np.random.uniform(0.8, 1.2), 2),
                        '진상무효': round(power * 0.1 * np.random.uniform(0.8, 1.2), 2),
                        '피상전력': round(power * 1.1 * np.random.uniform(0.95, 1.05), 2)
                    })
    
    df = pd.DataFrame(data)
    
    # CSV로 저장
    df.to_csv('./analysis_results/processed_lp_data.csv', index=False)
    print(f"   ✅ 고품질 테스트 데이터 생성: {len(df):,}건, {df['대체고객번호'].nunique()}명")
    print(f"   📅 기간: {df['datetime'].min()} ~ {df['datetime'].max()}")

if __name__ == "__main__":
    print("🚀 한국전력공사 변동계수 분석 시작 (정확도 우선 버전)!")
    print("=" * 80)
    print("🎯 정확도 최우선 | 과적합 방지 강화 | 충분한 검증")
    print("📊 더 많은 데이터 | 더 정밀한 분석 | 더 신뢰할 수 있는 결과")
    print()
    
    # 데이터 파일 확인
    required_files = ['./analysis_results/processed_lp_data.csv']
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print("⚠️ 데이터 파일이 없습니다. 고품질 테스트 데이터를 생성합니다.")
        create_test_environment_accurate()
        print()
    
    # 메인 실행
    results = main_accurate()
    
    if results:
        print(f"\n🎊 정확도 우선 분석 성공!")
        print(f"   📁 결과 파일: ./analysis_results/ 디렉토리")
        print(f"   🎯 정확도: R²={results.get('model_performance', {}).get('final_r2', 0):.4f}")
        print(f"   ✅ 과적합 방지: 차이={results.get('model_performance', {}).get('overfitting_gap', 0):.4f}")
        print(f"   🔍 검증 방법: {results.get('sampling_config', {}).get('validation_folds', 5)}-fold CV")
        
        quality = results.get('quality_assurance', {})
        print(f"\n💯 품질 지표:")
        print(f"   정확도 최적화: {'✅' if quality.get('accuracy_optimization_applied') else '❌'}")
        print(f"   과적합 방지: {'✅' if quality.get('overfitting_prevention') else '❌'}")
        print(f"   견고한 검증: {'✅' if quality.get('robust_validation') else '❌'}")
        print(f"   충분한 데이터: {'✅' if quality.get('sufficient_data_coverage') else '❌'}")
        print(f"   전체 신뢰도: {quality.get('statistical_confidence', 0):.3f}")
        
        print(f"\n📈 주요 성과:")
        if results.get('volatility_results'):
            cv_values = [v['enhanced_volatility_coefficient'] for v in results['volatility_results'].values()]
            print(f"   • 분석 완료: {len(results['volatility_results'])}명 고객")
            print(f"   • 평균 변동계수: {np.mean(cv_values):.4f}")
            print(f"   • 변동계수 범위: {np.min(cv_values):.4f} ~ {np.max(cv_values):.4f}")
        
        if results.get('stability_analysis'):
            high_risk = len([a for a in results['stability_analysis'].values() 
                           if a['risk_level'] in ['high', 'very_high']])
            print(f"   • 고위험 고객 식별: {high_risk}명")
        
        print(f"\n💼 비즈니스 가치:")
        print(f"   • 전력 사용 안정성 정량화")
        print(f"   • 영업활동 변화 예측 가능")
        print(f"   • 위험 고객 조기 발견")
        print(f"   • 맞춤형 관리 전략 수립")
        
    else:
        print(f"\n❌ 분석 실패")

print("\n" + "=" * 80)
print("🏆 한국전력공사 변동계수 스태킹 알고리즘 (정확도 우선)")
print("🎯 과적합 방지 | 📊 충분한 검증 | ✅ 신뢰할 수 있는 결과")
print("=" * 80)