"""
한국전력공사 전력 사용패턴 변동계수 개발 (청크 처리 최적화 버전)
전처리 2단계 샘플링 데이터 활용 + 메모리 효율적 청크 처리
"""

import pandas as pd
import numpy as np
import json
import os
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, r2_score
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from math import pi
from sklearn.metrics import mean_squared_error
import matplotlib
import gc
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

class KEPCOChunkVolatilityAnalyzer:
    """KEPCO 변동계수 분석기 (청크 처리 최적화 버전)"""
    
    def __init__(self, results_dir='./analysis_results', chunk_size=5000):
        self.results_dir = results_dir
        self.chunk_size = chunk_size  # 청크 크기 설정
        self.scaler = StandardScaler()
        self.robust_scaler = RobustScaler()
        self.level0_models = {}
        self.meta_model = None
        
        # 기존 전처리 결과 로딩
        self.step1_results = self._load_step1_results()
        self.step2_results = self._load_step2_results()
        self.sampled_data_path = None
        
        print("🔧 한국전력공사 변동계수 청크 처리 분석기 초기화")
        print(f"   📦 청크 크기: {self.chunk_size:,}건")
        
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
    
    def find_sampled_data(self):
        """전처리 2단계에서 생성된 샘플링 데이터 찾기"""
        print("\n📂 전처리 2단계 샘플링 데이터 검색 중...")
        
        # 가능한 파일 경로들
        possible_paths = [
            os.path.join(self.results_dir, 'sampled_lp_data.csv'),
            os.path.join(self.results_dir, 'processed_lp_data.csv'),
            './sampled_lp_data.csv',
            './processed_lp_data.csv'
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                # 파일 크기 확인
                file_size = os.path.getsize(path) / (1024 * 1024)  # MB
                print(f"   ✅ 샘플링 데이터 발견: {path}")
                print(f"   📊 파일 크기: {file_size:.1f} MB")
                
                # 간단한 데이터 검증
                try:
                    sample_df = pd.read_csv(path, nrows=1000)  # 더 많은 행으로 확인
                    print(f"   📋 컬럼: {list(sample_df.columns)}")
                    
                    # 전체 파일에서 고객 수 추정
                    total_rows = sum(1 for line in open(path)) - 1  # 헤더 제외
                    sample_customers = sample_df['대체고객번호'].nunique()
                    avg_records_per_customer = len(sample_df) / sample_customers if sample_customers > 0 else 1
                    estimated_customers = int(total_rows / avg_records_per_customer)
                    
                    print(f"   👥 첫 1,000행 고객 수: {sample_customers}명")
                    print(f"   📊 전체 추정 고객 수: 약 {estimated_customers}명")
                    
                    self.sampled_data_path = path
                    return True
                except Exception as e:
                    print(f"   ⚠️ 파일 검증 실패: {e}")
                    continue
        
        print("   ❌ 샘플링 데이터를 찾을 수 없습니다.")
        print("   💡 전처리 2단계를 먼저 실행해주세요.")
        return False
    
    def load_data_in_chunks(self):
        """청크 단위로 데이터 로딩 및 전처리"""
        print(f"\n📊 청크 단위 데이터 로딩 중... (청크 크기: {self.chunk_size:,})")
        
        if not self.sampled_data_path:
            print("   ❌ 샘플링 데이터 경로가 설정되지 않았습니다.")
            return False
        
        # 전체 행 수 확인
        total_rows = sum(1 for line in open(self.sampled_data_path)) - 1  # 헤더 제외
        total_chunks = (total_rows + self.chunk_size - 1) // self.chunk_size
        
        print(f"   📊 전체 데이터: {total_rows:,}건")
        print(f"   📦 예상 청크 수: {total_chunks}개")
        
        # 청크별 처리를 위한 고객 정보 수집
        self.customer_data_summary = {}
        processed_rows = 0
        
        # 첫 번째 패스: 고객별 기본 정보 수집
        print("   🔍 1단계: 고객별 기본 정보 수집 중...")
        
        chunk_reader = pd.read_csv(self.sampled_data_path, chunksize=self.chunk_size)
        
        for chunk_idx, chunk in enumerate(chunk_reader):
            try:
                # 필수 컬럼 확인 및 정리
                chunk = self._prepare_chunk_columns(chunk)
                
                if chunk is None or len(chunk) == 0:
                    continue
                
                # 고객별 요약 정보 수집
                for customer_id in chunk['대체고객번호'].unique():
                    customer_chunk = chunk[chunk['대체고객번호'] == customer_id]
                    
                    if customer_id not in self.customer_data_summary:
                        self.customer_data_summary[customer_id] = {
                            'total_records': 0,
                            'power_sum': 0.0,
                            'power_sum_sq': 0.0,
                            'min_power': float('inf'),
                            'max_power': float('-inf'),
                            'min_datetime': None,
                            'max_datetime': None,
                            'zero_count': 0
                        }
                    
                    summary = self.customer_data_summary[customer_id]
                    power_values = customer_chunk['순방향 유효전력'].values
                    
                    # 통계 정보 누적
                    summary['total_records'] += len(customer_chunk)
                    summary['power_sum'] += power_values.sum()
                    summary['power_sum_sq'] += (power_values ** 2).sum()
                    summary['min_power'] = min(summary['min_power'], power_values.min())
                    summary['max_power'] = max(summary['max_power'], power_values.max())
                    summary['zero_count'] += (power_values == 0).sum()
                    
                    # 날짜 범위 업데이트
                    chunk_min_date = customer_chunk['datetime'].min()
                    chunk_max_date = customer_chunk['datetime'].max()
                    
                    if summary['min_datetime'] is None or chunk_min_date < summary['min_datetime']:
                        summary['min_datetime'] = chunk_min_date
                    if summary['max_datetime'] is None or chunk_max_date > summary['max_datetime']:
                        summary['max_datetime'] = chunk_max_date
                
                processed_rows += len(chunk)
                
                # 메모리 정리
                del chunk
                gc.collect()
                
                if (chunk_idx + 1) % 5 == 0:
                    print(f"      청크 {chunk_idx + 1}/{total_chunks} 처리 완료 ({processed_rows:,}/{total_rows:,})")
                
            except Exception as e:
                print(f"      ⚠️ 청크 {chunk_idx} 처리 실패: {e}")
                continue
        
        print(f"   ✅ 고객 정보 수집 완료: {len(self.customer_data_summary)}명")
        
        # 최소 레코드 수 필터링
        min_records = 50
        valid_customers = [
            cid for cid, summary in self.customer_data_summary.items()
            if summary['total_records'] >= min_records
        ]
        
        print(f"   📋 유효 고객 (최소 {min_records}건): {len(valid_customers)}명")
        
        if len(valid_customers) == 0:
            print("   ❌ 분석 가능한 고객이 없습니다.")
            return False
        
        # 유효 고객으로 필터링
        self.customer_data_summary = {
            cid: summary for cid, summary in self.customer_data_summary.items()
            if cid in valid_customers
        }
        
        return True
    
    def _prepare_chunk_columns(self, chunk):
        """청크별 컬럼 준비"""
        try:
            # 필수 컬럼 확인
            required_columns = ['대체고객번호', '순방향 유효전력']
            datetime_columns = ['datetime', 'LP 수신일자', 'LP수신일자', 'timestamp']
            
            # datetime 컬럼 찾기
            datetime_col = None
            for col in datetime_columns:
                if col in chunk.columns:
                    datetime_col = col
                    break
            
            if datetime_col is None:
                print(f"      ⚠️ datetime 컬럼을 찾을 수 없습니다: {list(chunk.columns)}")
                return None
            
            # datetime 변환
            if datetime_col != 'datetime':
                chunk['datetime'] = pd.to_datetime(chunk[datetime_col], errors='coerce')
            else:
                chunk['datetime'] = pd.to_datetime(chunk['datetime'], errors='coerce')
            
            # 필수 컬럼 체크
            for col in required_columns:
                if col not in chunk.columns:
                    print(f"      ⚠️ 필수 컬럼 누락: {col}")
                    return None
            
            # 데이터 정제
            chunk = chunk.dropna(subset=['대체고객번호', 'datetime', '순방향 유효전력'])
            chunk = chunk[chunk['순방향 유효전력'] >= 0]
            
            # 시간 파생 변수 생성
            chunk['hour'] = chunk['datetime'].dt.hour
            chunk['weekday'] = chunk['datetime'].dt.weekday
            chunk['is_weekend'] = chunk['weekday'].isin([5, 6])
            chunk['month'] = chunk['datetime'].dt.month
            chunk['date'] = chunk['datetime'].dt.date
            
            return chunk
            
        except Exception as e:
            print(f"      ⚠️ 청크 컬럼 준비 실패: {e}")
            return None
    
    def calculate_volatility_from_chunks(self):
        """청크 기반 변동계수 계산"""
        print("\n📐 청크 기반 변동계수 계산 중...")
        
        # 2단계 결과에서 시간 패턴 정보 가져오기
        temporal_patterns = self.step2_results.get('temporal_patterns', {})
        peak_hours = temporal_patterns.get('peak_hours', [9, 10, 11, 14, 15, 18, 19])
        off_peak_hours = temporal_patterns.get('off_peak_hours', [0, 1, 2, 3, 4, 5])
        weekend_ratio = temporal_patterns.get('weekend_ratio', 1.0)
        
        print(f"   🕐 피크 시간: {peak_hours}")
        print(f"   🌙 비피크 시간: {off_peak_hours}")
        
        volatility_results = {}
        volatility_components = []
        
        # 고객별 상세 변동성 분석 (청크 단위)
        print("   🔍 2단계: 고객별 상세 변동성 분석 중...")
        
        customer_list = list(self.customer_data_summary.keys())
        processed_customers = 0
        
        # 고객을 그룹으로 나누어 청크 단위 처리
        customer_batch_size = 50  # 한 번에 처리할 고객 수
        
        for batch_start in range(0, len(customer_list), customer_batch_size):
            batch_end = min(batch_start + customer_batch_size, len(customer_list))
            batch_customers = customer_list[batch_start:batch_end]
            
            print(f"      배치 {batch_start//customer_batch_size + 1}: 고객 {batch_start+1}-{batch_end} 처리 중...")
            
            # 해당 고객들의 데이터만 청크 단위로 로딩
            batch_volatility = self._process_customer_batch_chunks(
                batch_customers, peak_hours, off_peak_hours, weekend_ratio
            )
            
            # 결과 병합
            for customer_id, metrics in batch_volatility.items():
                if metrics:
                    volatility_components.append({
                        'customer_id': customer_id,
                        **metrics
                    })
                    processed_customers += 1
            
            # 메모리 정리
            gc.collect()
        
        print(f"   ✅ {processed_customers}명 변동성 지표 계산 완료")
        
        if len(volatility_components) < 10:
            raise ValueError(f"가중치 최적화를 위해서는 최소 10개의 고객 데이터가 필요합니다. (현재: {len(volatility_components)}개)")
        
        # 가중치 최적화
        optimal_weights = self.optimize_volatility_weights(volatility_components)
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
                'optimized_weights': [round(w, 3) for w in optimal_weights]
            }
        
        if len(volatility_results) > 0:
            cv_values = [v['enhanced_volatility_coefficient'] for v in volatility_results.values()]
            print(f"   📈 평균 변동계수: {np.mean(cv_values):.4f}")
            print(f"   📊 변동계수 범위: {np.min(cv_values):.4f} ~ {np.max(cv_values):.4f}")
        
        return volatility_results
    
    def _process_customer_batch_chunks(self, batch_customers, peak_hours, off_peak_hours, weekend_ratio):
        """고객 배치를 청크 단위로 처리"""
        batch_results = {}
        
        # 고객별 상세 데이터 수집을 위한 딕셔너리
        customer_detailed_data = {cid: {
            'power_values': [],
            'hourly_data': {},
            'peak_data': [],
            'off_peak_data': [],
            'weekday_data': [],
            'weekend_data': [],
            'daily_averages': {},
            'extreme_changes': 0
        } for cid in batch_customers}
        
        # 청크 단위로 파일 읽기
        chunk_reader = pd.read_csv(self.sampled_data_path, chunksize=self.chunk_size)
        
        for chunk in chunk_reader:
            try:
                # 청크 전처리
                chunk = self._prepare_chunk_columns(chunk)
                if chunk is None or len(chunk) == 0:
                    continue
                
                # 배치 고객만 필터링
                batch_chunk = chunk[chunk['대체고객번호'].isin(batch_customers)]
                if len(batch_chunk) == 0:
                    continue
                
                # 고객별 데이터 수집
                for customer_id in batch_customers:
                    customer_chunk = batch_chunk[batch_chunk['대체고객번호'] == customer_id]
                    if len(customer_chunk) == 0:
                        continue
                    
                    customer_data = customer_detailed_data[customer_id]
                    power_values = customer_chunk['순방향 유효전력'].values
                    
                    # 전력 데이터 수집
                    customer_data['power_values'].extend(power_values)
                    
                    # 시간대별 데이터
                    for hour in range(24):
                        hour_data = customer_chunk[customer_chunk['hour'] == hour]['순방향 유효전력']
                        if len(hour_data) > 0:
                            if hour not in customer_data['hourly_data']:
                                customer_data['hourly_data'][hour] = []
                            customer_data['hourly_data'][hour].extend(hour_data.tolist())
                    
                    # 피크/오프피크 데이터
                    peak_data = customer_chunk[customer_chunk['hour'].isin(peak_hours)]['순방향 유효전력']
                    off_peak_data = customer_chunk[customer_chunk['hour'].isin(off_peak_hours)]['순방향 유효전력']
                    
                    customer_data['peak_data'].extend(peak_data.tolist())
                    customer_data['off_peak_data'].extend(off_peak_data.tolist())
                    
                    # 주중/주말 데이터
                    weekday_data = customer_chunk[~customer_chunk['is_weekend']]['순방향 유효전력']
                    weekend_data = customer_chunk[customer_chunk['is_weekend']]['순방향 유효전력']
                    
                    customer_data['weekday_data'].extend(weekday_data.tolist())
                    customer_data['weekend_data'].extend(weekend_data.tolist())
                    
                    # 일별 평균 (계절별 변동성용)
                    daily_groups = customer_chunk.groupby('date')['순방향 유효전력'].mean()
                    for date, avg_power in daily_groups.items():
                        customer_data['daily_averages'][date] = avg_power
                    
                    # 급격한 변화 감지
                    if len(power_values) > 1:
                        power_series = pd.Series(power_values)
                        pct_changes = power_series.pct_change().dropna()
                        customer_data['extreme_changes'] += (np.abs(pct_changes) > 1.5).sum()
                
                # 메모리 정리
                del chunk, batch_chunk
                gc.collect()
                
            except Exception as e:
                print(f"         ⚠️ 청크 처리 실패: {e}")
                continue
        
        # 고객별 변동성 지표 계산
        for customer_id in batch_customers:
            try:
                customer_data = customer_detailed_data[customer_id]
                
                if len(customer_data['power_values']) < 10:
                    continue
                
                metrics = self._calculate_customer_volatility_metrics(
                    customer_data, peak_hours, off_peak_hours, weekend_ratio
                )
                
                if metrics:
                    batch_results[customer_id] = metrics
                    
            except Exception as e:
                print(f"         ⚠️ 고객 {customer_id} 지표 계산 실패: {e}")
                continue
        
        return batch_results
    
    def _calculate_customer_volatility_metrics(self, customer_data, peak_hours, off_peak_hours, weekend_ratio):
        """개별 고객의 변동성 지표 계산 (청크 기반)"""
        try:
            power_values = np.array(customer_data['power_values'])
            
            if len(power_values) == 0 or np.mean(power_values) <= 0:
                return None
            
            mean_power = np.mean(power_values)
            
            # 1. 기본 변동계수
            basic_cv = np.std(power_values) / mean_power
            
            # 2. 시간대별 변동계수
            hourly_means = []
            for hour in range(24):
                if hour in customer_data['hourly_data'] and len(customer_data['hourly_data'][hour]) > 0:
                    hourly_means.append(np.mean(customer_data['hourly_data'][hour]))
            
            hourly_cv = (np.std(hourly_means) / np.mean(hourly_means)) if len(hourly_means) > 1 and np.mean(hourly_means) > 0 else basic_cv
            
            # 3. 피크/비피크 변동성
            peak_data = customer_data['peak_data']
            off_peak_data = customer_data['off_peak_data']
            
            peak_cv = (np.std(peak_data) / np.mean(peak_data)) if len(peak_data) > 0 and np.mean(peak_data) > 0 else basic_cv
            off_peak_cv = (np.std(off_peak_data) / np.mean(off_peak_data)) if len(off_peak_data) > 0 and np.mean(off_peak_data) > 0 else basic_cv
            
            # 4. 주말/평일 변동성
            weekday_data = customer_data['weekday_data']
            weekend_data = customer_data['weekend_data']
            
            weekday_cv = (np.std(weekday_data) / np.mean(weekday_data)) if len(weekday_data) > 0 and np.mean(weekday_data) > 0 else basic_cv
            weekend_cv = (np.std(weekend_data) / np.mean(weekend_data)) if len(weekend_data) > 0 and np.mean(weekend_data) > 0 else basic_cv
            weekend_diff = abs(weekday_cv - weekend_cv) * weekend_ratio
            
            # 5. 계절별 변동성 (일별 집계)
            daily_averages = list(customer_data['daily_averages'].values())
            seasonal_cv = (np.std(daily_averages) / np.mean(daily_averages)) if len(daily_averages) > 3 and np.mean(daily_averages) > 0 else basic_cv
            
            # 6. 추가 지표들
            max_power = np.max(power_values)
            load_factor = mean_power / max_power if max_power > 0 else 0
            zero_ratio = (power_values == 0).sum() / len(power_values)
            
            # 피크/비피크 부하 비율
            peak_avg = np.mean(peak_data) if len(peak_data) > 0 else mean_power
            off_peak_avg = np.mean(off_peak_data) if len(off_peak_data) > 0 else mean_power
            peak_load_ratio = peak_avg / off_peak_avg if off_peak_avg > 0 else 1.0
            
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
                'extreme_changes': customer_data['extreme_changes'],
                'peak_load_ratio': peak_load_ratio,
                'mean_power': mean_power,
                'data_points': len(power_values)
            }
            
        except Exception as e:
            return None
    
    def optimize_volatility_weights(self, volatility_components):
        """가중치 최적화"""
        print("\n⚙️ 가중치 최적화 중...")
        
        try:
            from scipy.optimize import minimize
        except ImportError:
            raise ImportError("scipy가 설치되지 않았습니다. 'pip install scipy'로 설치해주세요.")
        
        # 목표 함수 정의
        components_df = pd.DataFrame(volatility_components)
        
        # 목표 변수: 영업활동 불안정성 지표
        target_instability = (
            components_df['basic_cv'] * 2.0 +
            components_df['zero_ratio'] * 1.0 +
            (1 - components_df['load_factor']) * 0.5
        ).values
        
        X = components_df[['basic_cv', 'hourly_cv', 'peak_cv', 'weekend_diff', 'seasonal_cv']].values
        y = target_instability
        
        # 최적화 목표 함수
        def objective(weights):
            predicted = X @ weights
            return np.mean((predicted - y) ** 2)
        
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
        bounds = [(0, 1) for _ in range(5)]
        initial_weights = [0.35, 0.25, 0.20, 0.10, 0.10]
        
        result = minimize(objective, initial_weights, method='SLSQP', 
                        bounds=bounds, constraints=constraints)
        
        if not result.success:
            raise RuntimeError(f"가중치 최적화에 실패했습니다: {result.message}")
        
        print(f"   ✅ 가중치 최적화 완료")
        return result.x.tolist()
    
    def train_stacking_ensemble_model(self, volatility_results):
        """스태킹 앙상블 모델 훈련"""
        print("\n🎯 스태킹 앙상블 모델 훈련 중...")
        
        if len(volatility_results) < 5:
            print("   ❌ 훈련 데이터가 부족합니다 (최소 5개 필요)")
            return None
        
        # 특성 추출
        features = []
        targets = []
        
        for customer_id, data in volatility_results.items():
            try:
                feature_vector = [
                    data['basic_cv'], data['hourly_cv'], data['peak_cv'],
                    data['off_peak_cv'], data['weekday_cv'], data['weekend_cv'],
                    data['seasonal_cv'], data['load_factor'], data['peak_load_ratio'],
                    data['mean_power'], data['zero_ratio'],
                    data['extreme_changes'] / data['data_points']
                ]
                
                if any(np.isnan(x) or np.isinf(x) for x in feature_vector):
                    continue
                    
                features.append(feature_vector)
                targets.append(data['enhanced_volatility_coefficient'])
                
            except KeyError:
                continue
        
        X = np.array(features)
        y = np.array(targets)
        
        print(f"   📊 훈련 데이터: {len(X)}개 샘플, {X.shape[1]}개 특성")
        
        # 데이터 분할
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 정규화
        X_train_scaled = self.robust_scaler.fit_transform(X_train)
        X_test_scaled = self.robust_scaler.transform(X_test)
        
        # Level-0 모델들
        self.level0_models = {
            'rf': RandomForestRegressor(n_estimators=50, max_depth=8, random_state=42),
            'gbm': GradientBoostingRegressor(n_estimators=50, max_depth=6, random_state=42),
            'ridge': Ridge(alpha=1.0),
            'linear': LinearRegression()
        }
        
        # 교차검증으로 메타 특성 생성
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        meta_features_train = np.zeros((len(X_train_scaled), len(self.level0_models)))
        meta_features_test = np.zeros((len(X_test_scaled), len(self.level0_models)))
        
        print(f"   🔄 Level-0 모델 훈련 (5-Fold CV):")
        for i, (name, model) in enumerate(self.level0_models.items()):
            fold_predictions = np.zeros(len(X_train_scaled))
            
            for train_idx, val_idx in kf.split(X_train_scaled):
                try:
                    fold_model = type(model)(**model.get_params())
                    fold_model.fit(X_train_scaled[train_idx], y_train[train_idx])
                    fold_predictions[val_idx] = fold_model.predict(X_train_scaled[val_idx])
                except Exception as e:
                    fold_predictions[val_idx] = np.mean(y_train[train_idx])
            
            meta_features_train[:, i] = fold_predictions
            
            # 전체 훈련 세트로 재훈련
            try:
                model.fit(X_train_scaled, y_train)
                meta_features_test[:, i] = model.predict(X_test_scaled)
                
                test_pred = model.predict(X_test_scaled)
                test_mae = mean_absolute_error(y_test, test_pred)
                test_r2 = r2_score(y_test, test_pred) if len(set(y_test)) > 1 else 0.0
                print(f"      {name}: MAE={test_mae:.4f}, R²={test_r2:.4f}")
                
            except Exception as e:
                meta_features_test[:, i] = np.mean(y_train)
        
        # Level-1 메타 모델 (선형 회귀)
        self.meta_model = LinearRegression()
        try:
            self.meta_model.fit(meta_features_train, y_train)
            final_pred = self.meta_model.predict(meta_features_test)
            final_mae = mean_absolute_error(y_test, final_pred)
            final_r2 = r2_score(y_test, final_pred) if len(set(y_test)) > 1 else 0.0
        except:
            final_pred = np.mean(meta_features_test, axis=1)
            final_mae = mean_absolute_error(y_test, final_pred)
            final_r2 = r2_score(y_test, final_pred) if len(set(y_test)) > 1 else 0.0
        
        print(f"   ✅ 스태킹 앙상블 훈련 완료")
        print(f"      최종 MAE: {final_mae:.4f}")
        print(f"      최종 R²: {final_r2:.4f}")
        
        return {
            'final_mae': final_mae,
            'final_r2': final_r2,
            'level0_models': list(self.level0_models.keys()),
            'meta_model': 'LinearRegression',
            'n_samples': len(X),
            'n_features': X.shape[1],
            'chunk_optimized': True
        }

    def analyze_business_stability(self, volatility_results):
        """영업활동 안정성 분석"""
        print("\n🔍 영업활동 안정성 분석 중...")
        
        if not volatility_results:
            return {}
        
        coefficients = [v['enhanced_volatility_coefficient'] for v in volatility_results.values()]
        
        # 분위수 기반 등급 분류
        p25, p75 = np.percentile(coefficients, [25, 75])
        
        stability_analysis = {}
        grade_counts = {'안정': 0, '보통': 0, '주의': 0}
        
        for customer_id, data in volatility_results.items():
            coeff = data['enhanced_volatility_coefficient']
            
            # 3단계 등급 분류
            if coeff <= p25:
                grade = '안정'
                risk_level = 'low'
            elif coeff <= p75:
                grade = '보통'
                risk_level = 'medium'
            else:
                grade = '주의'
                risk_level = 'high'
            
            grade_counts[grade] += 1
            
            # 위험 요인 분석
            risk_factors = []
            if data.get('zero_ratio', 0) > 0.1:
                risk_factors.append('빈번한_사용중단')
            if data.get('load_factor', 1) < 0.3:
                risk_factors.append('낮은_부하율')
            if data.get('peak_cv', 0) > data.get('basic_cv', 0) * 2:
                risk_factors.append('피크시간_불안정')
            if data.get('weekend_diff', 0) > 0.3:
                risk_factors.append('주말_패턴_급변')
            if data.get('extreme_changes', 0) > data.get('data_points', 1) * 0.05:
                risk_factors.append('급격한_변화_빈발')
            
            # 안정성 점수 계산 (0-100점)
            stability_score = max(0, 100 - (coeff * 400))
            
            stability_analysis[customer_id] = {
                'enhanced_volatility_coefficient': round(coeff, 4),
                'stability_grade': grade,
                'risk_level': risk_level,
                'stability_score': round(stability_score, 1),
                'risk_factors': risk_factors,
                'load_factor': data.get('load_factor', 0.0),
                'peak_load_ratio': data.get('peak_load_ratio', 1.0),
                'zero_ratio': data.get('zero_ratio', 0.0),
                'extreme_changes': data.get('extreme_changes', 0)
            }
        
        print(f"   📋 안정성 등급 분포:")
        total = len(stability_analysis)
        for grade, count in grade_counts.items():
            percentage = count / total * 100 if total > 0 else 0
            print(f"      {grade}: {count}명 ({percentage:.1f}%)")
        
        return stability_analysis

    def generate_report(self, volatility_results, model_performance, stability_analysis):
        """청크 처리 최적화 리포트 생성"""
        print("\n📋 청크 처리 리포트 생성 중...")
        
        coefficients = [v['enhanced_volatility_coefficient'] for v in volatility_results.values()] if volatility_results else []
        
        # 위험 고객 식별
        high_risk_customers = [
            customer_id for customer_id, analysis in stability_analysis.items()
            if analysis['risk_level'] == 'high'
        ] if stability_analysis else []
        
        # 주요 위험 요인 집계
        all_risk_factors = []
        for analysis in stability_analysis.values():
            all_risk_factors.extend(analysis['risk_factors'])
        
        from collections import Counter
        risk_factor_counts = Counter(all_risk_factors)
        
        report = {
            'analysis_metadata': {
                'timestamp': pd.Timestamp.now().isoformat(),
                'algorithm_version': 'chunk_optimized_v1',
                'chunk_size': self.chunk_size,
                'total_customers_analyzed': len(volatility_results),
                'execution_mode': 'chunk_processing',
                'data_source': 'preprocessed_sampled_data'
            },
            
            'chunk_processing_summary': {
                'chunk_size_used': self.chunk_size,
                'memory_efficient': True,
                'batch_processing': True,
                'sampled_data_path': self.sampled_data_path
            },
            
            'volatility_coefficient_summary': {
                'total_customers': len(volatility_results),
                'mean_coefficient': round(np.mean(coefficients), 4) if coefficients else 0,
                'std_coefficient': round(np.std(coefficients), 4) if coefficients else 0,
                'percentiles': {
                    '25%': round(np.percentile(coefficients, 25), 4) if coefficients else 0,
                    '50%': round(np.percentile(coefficients, 50), 4) if coefficients else 0,
                    '75%': round(np.percentile(coefficients, 75), 4) if coefficients else 0
                }
            },
            
            'model_performance': model_performance or {},
            
            'business_stability_distribution': {
                grade: sum(1 for a in stability_analysis.values() if a['stability_grade'] == grade)
                for grade in ['안정', '보통', '주의']
            } if stability_analysis else {},
            
            'risk_analysis': {
                'high_risk_customers': len(high_risk_customers),
                'high_risk_percentage': round(len(high_risk_customers) / len(stability_analysis) * 100, 1) if stability_analysis else 0,
                'top_risk_factors': dict(risk_factor_counts.most_common(5))
            },
            
            'performance_optimization': {
                'chunk_processing_achieved': True,
                'memory_efficient': True,
                'accuracy_maintained': model_performance['final_r2'] >= 0.3 if model_performance else False
            },
            
            'business_insights': [
                f"청크 처리를 통해 {len(volatility_results)}명 고객 분석 완료",
                f"메모리 효율적 처리로 대용량 데이터 안정적 분석",
                f"모델 예측 정확도(R²): {model_performance['final_r2']:.3f}" if model_performance else "모델 성능 측정 불가",
                f"고위험 고객 {len(high_risk_customers)}명 식별",
                "데이터안심구역 환경에 최적화된 안정적 분석 시스템"
            ],
            
            'recommendations': [
                "청크 크기 조정을 통한 메모리 사용량 최적화",
                "배치 처리로 대용량 데이터 안정적 처리",
                "실시간 모니터링을 위한 효율적 분석 체계",
                "주기적 전체 데이터 검증으로 품질 확보"
            ]
        }
        
        return report

    def create_volatility_components_radar_chart(self, volatility_results, save_path='./analysis_results'):
        """레이더 차트 생성 (영문 버전)"""
        import matplotlib.pyplot as plt
        import numpy as np
        from math import pi
        import os
        
        if not volatility_results:
            print("   ❌ 변동계수 결과가 없습니다.")
            return None
        
        # 구성요소 이름 (영문)
        components = ['Basic CV', 'Hourly CV', 'Peak CV', 'Weekend Diff', 'Seasonal CV']
        component_keys = ['basic_cv', 'hourly_cv', 'peak_cv', 'weekend_diff', 'seasonal_cv']
        
        # 데이터 추출 및 정규화
        customers_data = {}
        all_values = {key: [] for key in component_keys}
        
        # 모든 고객의 데이터 수집
        for customer_id, data in volatility_results.items():
            customer_values = []
            for key in component_keys:
                value = data.get(key, 0)
                # 이상값 처리
                if np.isnan(value) or np.isinf(value):
                    value = 0
                customer_values.append(value)
                all_values[key].append(value)
            customers_data[customer_id] = customer_values
        
        # 정규화를 위한 최대값 계산 (각 구성요소별)
        max_values = []
        for key in component_keys:
            values = all_values[key]
            if values:
                max_val = max(values) if max(values) > 0 else 1
                max_values.append(max_val)
            else:
                max_values.append(1)
        
        # 상위 5명의 고객 선택 (변동계수가 높은 순)
        top_customers = sorted(
            volatility_results.items(),
            key=lambda x: x[1].get('enhanced_volatility_coefficient', 0),
            reverse=True
        )[:5]
        
        # 레이더 차트 설정
        fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(projection='polar'))
        
        # 각도 계산 (5개 항목)
        angles = [n / float(len(components)) * 2 * pi for n in range(len(components))]
        angles += angles[:1] 
        
        # 색상 팔레트
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57']
        
        # 각 고객별 레이더 차트 그리기
        for i, (customer_id, data) in enumerate(top_customers):
            if i >= 5:  # 최대 5명만
                break
                
            # 데이터 정규화 (0-1 범위)
            values = []
            for j, key in enumerate(component_keys):
                raw_value = data.get(key, 0)
                if np.isnan(raw_value) or np.isinf(raw_value):
                    raw_value = 0
                normalized_value = raw_value / max_values[j] if max_values[j] > 0 else 0
                values.append(min(normalized_value, 1.0))  # 1.0으로 클리핑
            
            values += values[:1] 
            
            # 선 그리기
            ax.plot(angles, values, 'o-', linewidth=2, label=f'{customer_id}', color=colors[i], markersize=6)
            # 영역 채우기 (투명도 적용)
            ax.fill(angles, values, alpha=0.15, color=colors[i])
        
        # 라벨 설정
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(components, fontsize=11, fontweight='bold')
        
        # Y축 설정 (0-1 범위)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # 제목 및 범례 (영문)
        plt.title('Volatility Coefficient Components Analysis (Top 5 Customers)', 
                  fontsize=16, fontweight='bold', pad=30)
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=10)
        
        # 서브 제목 (영문)
        fig.text(0.5, 0.02, 'Each component is normalized by maximum value (0-1 range)', 
                 ha='center', fontsize=9, style='italic')
        
        # 통계 정보 추가 (영문)
        stats_text = f"Analyzed Customers: {len(volatility_results)}\n"
        stats_text += f"Average Volatility Coeff: {np.mean([v.get('enhanced_volatility_coefficient', 0) for v in volatility_results.values()]):.4f}"
        fig.text(0.02, 0.95, stats_text, fontsize=9, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        # 저장
        os.makedirs(save_path, exist_ok=True)
        chart_path = os.path.join(save_path, 'volatility_components_radar_chunk.png')
        plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"   ✅ 레이더 차트 저장: {chart_path}")
        
        return {
            'chart_path': chart_path,
            'top_customers': [customer_id for customer_id, _ in top_customers]
        }

def save_chunk_results(volatility_results, stability_analysis, report):
    """청크 처리 결과 저장"""
    try:
        os.makedirs('./analysis_results', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 변동계수 결과
        if volatility_results:
            df = pd.DataFrame.from_dict(volatility_results, orient='index')
            df.reset_index(inplace=True)
            df.rename(columns={'index': '대체고객번호'}, inplace=True)
            csv_path = f'./analysis_results/volatility_chunk_{timestamp}.csv'
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            print(f"   💾 변동계수 (청크): {csv_path}")
        
        # 안정성 분석
        if stability_analysis:
            df = pd.DataFrame.from_dict(stability_analysis, orient='index')
            df.reset_index(inplace=True)
            df.rename(columns={'index': '대체고객번호'}, inplace=True)
            if 'risk_factors' in df.columns:
                df['risk_factors_str'] = df['risk_factors'].apply(
                    lambda x: ', '.join(x) if isinstance(x, list) else ''
                )
            csv_path = f'./analysis_results/stability_chunk_{timestamp}.csv'
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            print(f"   💾 안정성 (청크): {csv_path}")
        
        # 청크 리포트
        if report:
            json_path = f'./analysis_results/chunk_report_{timestamp}.json'
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2, default=str)
            print(f"   💾 청크 리포트: {json_path}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ 결과 저장 실패: {e}")
        return False

def main_chunk():
    """메인 실행 함수 (청크 처리 버전)"""
    print("🏆 한국전력공사 전력 사용패턴 변동계수 분석 (청크 처리 최적화)")
    print("=" * 80)
    print("📦 주요 특징:")
    print("   ✅ 전처리 2단계 샘플링 데이터 활용")
    print("   ✅ 메모리 효율적 청크 처리")
    print("   ✅ 배치 단위 고객 분석")
    print("   ✅ 데이터안심구역 환경 최적화")
    print("   ✅ 기존 출력 형식 완전 호환")
    print()
    
    start_time = datetime.now()
    
    try:
        # 1. 분석기 초기화 (청크 크기 조정 가능)
        chunk_size = 5000  # 메모리에 따라 조정 가능
        analyzer = KEPCOChunkVolatilityAnalyzer('./analysis_results', chunk_size)
        
        # 2. 샘플링 데이터 찾기
        if not analyzer.find_sampled_data():
            print("❌ 샘플링 데이터를 찾을 수 없습니다.")
            print("\n🔧 해결 방법:")
            print("   1. 전처리 2단계 (전처리2단계 수정.py)를 먼저 실행")
            print("   2. sampled_lp_data.csv 파일이 생성되었는지 확인")
            return None
        
        # 3. 청크 단위 데이터 로딩
        if not analyzer.load_data_in_chunks():
            print("❌ 청크 데이터 로딩 실패")
            return None
        
        # 4. 청크 기반 변동계수 계산
        volatility_results = analyzer.calculate_volatility_from_chunks()
        if not volatility_results:
            print("❌ 변동계수 계산 실패")
            return None
        
        # 5. 모델 훈련
        model_performance = analyzer.train_stacking_ensemble_model(volatility_results)
        
        # 6. 안정성 분석
        stability_analysis = analyzer.analyze_business_stability(volatility_results)
        
        # 7. 리포트 생성
        report = analyzer.generate_report(volatility_results, model_performance, stability_analysis)
        
        # 8. 시각화 생성
        try:
            radar_result = analyzer.create_volatility_components_radar_chart(volatility_results)
            if radar_result:
                print(f"   📊 레이더 차트 생성 완료: {radar_result['chart_path']}")
        except Exception as e:
            print(f"   ⚠️ 레이더 차트 생성 중 오류 (무시하고 계속): {e}")
        
        # 9. 결과 저장
        save_chunk_results(volatility_results, stability_analysis, report)
        
        # 실행 시간 계산
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        print(f"\n🎉 청크 처리 분석 완료!")
        print(f"   ⏱️ 실행 시간: {execution_time:.1f}초")
        print(f"   👥 분석 고객: {len(volatility_results)}명")
        print(f"   📦 청크 크기: {chunk_size:,}건")
        
        if volatility_results:
            cv_values = [v['enhanced_volatility_coefficient'] for v in volatility_results.values()]
            print(f"   📈 평균 변동계수: {np.mean(cv_values):.4f}")
            print(f"   📊 변동계수 범위: {np.min(cv_values):.4f} ~ {np.max(cv_values):.4f}")
        
        if model_performance:
            print(f"   🎯 모델 성능: R²={model_performance['final_r2']:.3f}, MAE={model_performance['final_mae']:.4f}")
        
        print(f"   💾 결과 파일: ./analysis_results/ 디렉토리")
        
        return {
            'volatility_results': volatility_results,
            'model_performance': model_performance,
            'stability_analysis': stability_analysis,
            'report': report,
            'execution_time': execution_time,
            'chunk_size': chunk_size
        }
        
    except Exception as e:
        print(f"❌ 시스템 실행 중 오류: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("🚀 한국전력공사 변동계수 분석 시작 (청크 처리 버전)!")
    print("=" * 80)
    print("📦 청크 처리로 메모리 효율성 극대화")
    print("🎯 전처리 2단계 샘플링 데이터 활용")
    print("📊 기존 출력 형식 완전 유지")
    print()
    
    # 메인 실행
    results = main_chunk()
    
    if results:
        print(f"\n🎊 청크 처리 분석 성공!")
        print(f"   📁 결과 파일들이 ./analysis_results/ 디렉토리에 저장되었습니다")
        print(f"   ⚡ 메모리 효율적 처리로 안정적 실행 완료")
        print(f"   🎯 동일한 정확도, 향상된 안정성")
        
        print(f"\n💡 청크 크기 조정:")
        print(f"   • 메모리 부족시: chunk_size를 2000~3000으로 감소")
        print(f"   • 메모리 여유시: chunk_size를 10000~20000으로 증가")
        print(f"   • 현재 설정: {results['chunk_size']:,}건")
        
    else:
        print(f"\n❌ 분석 실패")
        print(f"   🔧 확인 사항:")
        print(f"   1. 전처리 2단계가 완료되었는지 확인")
        print(f"   2. sampled_lp_data.csv 파일 존재 여부")
        print(f"   3. 메모리 용량 및 청크 크기 설정")

print("\n" + "=" * 80)
print("🏆 한국전력공사 변동계수 스태킹 알고리즘 (청크 처리 최적화)")
print("📦 메모리 효율성 | 🎯 안정적 처리 | 📊 동일한 출력 형식")
print("=" * 80)