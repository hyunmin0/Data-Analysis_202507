"""
한국전력공사 전력 사용패턴 변동계수 개발 (기존 전처리 결과 활용)
목표: 기업의 전력 사용 안정성과 영업활동 변화 예측

입력 데이터:
1. analysis_results.json (1단계 전처리 결과)
2. analysis_results2.json (2단계 시계열 분석)
3. processed_lp_data.h5 (전처리된 LP 데이터)
4. 한전_통합데이터.xlsx (한전 공공데이터)
"""

import pandas as pd
import numpy as np
import json
import os
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class KEPCOStackingVolatilityAnalyzer:
    """한국전력공사 전력 사용패턴 변동계수 스태킹 분석기 (전처리 결과 활용)"""
    
    def __init__(self, results_dir='./analysis_results'):
        self.results_dir = results_dir
        self.scaler = StandardScaler()
        self.level0_models = {}
        self.meta_model = None
        
        # 기존 전처리 결과 로딩
        self.step1_results = self._load_step1_results()
        self.step2_results = self._load_step2_results()
        self.lp_data = None
        self.kepco_data = None
        
        print("🔧 한국전력공사 변동계수 스태킹 분석기 초기화")
        print(f"   📁 결과 디렉토리: {results_dir}")
        
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
                print(f"   ⚠️ 1단계 결과 파일 없음: {file_path}")
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
                print(f"   ⚠️ 2단계 결과 파일 없음: {file_path}")
                return {}
        except Exception as e:
            print(f"   ❌ 2단계 결과 로딩 실패: {e}")
            return {}
    
    def load_preprocessed_data(self):
        """전처리된 데이터 로딩"""
        print("\n📊 전처리된 데이터 로딩 중...")
        
        # 1. LP 데이터 로딩 (HDF5 우선)
        hdf5_path = os.path.join(self.results_dir, 'processed_lp_data.h5')
        csv_path = os.path.join(self.results_dir, 'processed_lp_data.csv')
        
        if os.path.exists(hdf5_path):
            try:
                self.lp_data = pd.read_hdf(hdf5_path, key='df')
                print(f"   ✅ HDF5 LP 데이터: {len(self.lp_data):,}건")
                loading_method = "HDF5"
            except Exception as e:
                print(f"   ⚠️ HDF5 로딩 실패: {e}")
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
        
        # datetime 컬럼 처리
        if 'datetime' in self.lp_data.columns:
            self.lp_data['datetime'] = pd.to_datetime(self.lp_data['datetime'])
        elif 'LP 수신일자' in self.lp_data.columns:
            self.lp_data['datetime'] = pd.to_datetime(self.lp_data['LP 수신일자'])
        
        print(f"   📈 로딩 방법: {loading_method}")
        print(f"   📅 기간: {self.lp_data['datetime'].min()} ~ {self.lp_data['datetime'].max()}")
        print(f"   👥 고객수: {self.lp_data['대체고객번호'].nunique()}")
        
        # 2. 한전 통합 데이터 로딩
        kepco_path = '한전_통합데이터.xlsx'
        if os.path.exists(kepco_path):
            try:
                self.kepco_data = pd.read_excel(kepco_path, sheet_name='전체데이터')
                print(f"   ✅ 한전 통합 데이터: {len(self.kepco_data):,}건")
            except Exception as e:
                print(f"   ⚠️ 한전 데이터 로딩 실패: {e}")
                self.kepco_data = None
        else:
            print(f"   ⚠️ 한전 통합 데이터 없음: {kepco_path}")
            self.kepco_data = None
        
        return True
    
    def optimize_volatility_weights(self, volatility_components):
        """데이터 기반 가중치 최적화"""
        print("\n⚙️ 변동계수 가중치 최적화 중...")
        
        if len(volatility_components) < 20:
            print("   ⚠️ 최적화에 충분한 데이터가 없습니다. 기본 가중치 사용")
            return [0.35, 0.25, 0.20, 0.10, 0.10]
        
        from scipy.optimize import minimize
        import warnings
        warnings.filterwarnings('ignore')
        
        # 성분별 데이터 준비
        components_df = pd.DataFrame(volatility_components)
        
        # 목표 변수 생성 (영업활동 변화의 대리 지표)
        # 방법 1: 높은 변동성 = 높은 위험도
        # 방법 2: 실제 영업 지표가 있다면 활용 (매출, 계약 변경 등)
        
        # 여기서는 종합적인 불안정성을 목표로 설정
        target_instability = []
        
        for idx, row in components_df.iterrows():
            # 복합 불안정성 지표 계산
            instability = (
                row['basic_cv'] * 2.0 +           # 기본 변동성 높으면 불안정
                row['extreme_changes'] * 0.01 +   # 급격한 변화 많으면 불안정  
                row['zero_ratio'] * 1.0 +         # 사용 중단 많으면 불안정
                (1 - row['load_factor']) * 0.5    # 부하율 낮으면 불안정
            )
            target_instability.append(instability)
        
        X = components_df[['basic_cv', 'hourly_cv', 'peak_cv', 'weekend_diff', 'seasonal_cv']].values
        y = np.array(target_instability)
        
        # 제약 조건: 가중치 합 = 1, 모든 가중치 >= 0
        def objective(weights):
            predicted = X @ weights
            return np.mean((predicted - y) ** 2)  # MSE
        
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},  # 합 = 1
        ]
        
        bounds = [(0, 1) for _ in range(5)]  # 0 <= weight <= 1
        
        # 초기값 (기존 가중치)
        initial_weights = [0.35, 0.25, 0.20, 0.10, 0.10]
        
        try:
            result = minimize(
                objective, 
                initial_weights, 
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
            if result.success:
                optimized_weights = result.x
                improvement = objective(initial_weights) - objective(optimized_weights)
                
                print(f"   ✅ 가중치 최적화 완료")
                print(f"   📊 기존 가중치: {[round(w, 3) for w in initial_weights]}")
                print(f"   🎯 최적 가중치: {[round(w, 3) for w in optimized_weights]}")
                print(f"   📈 개선도: {improvement:.4f}")
                
                return optimized_weights.tolist()
            else:
                print(f"   ⚠️ 최적화 실패: {result.message}")
                return initial_weights
                
        except Exception as e:
            print(f"   ❌ 최적화 오류: {e}")
            return initial_weights
        """향상된 변동계수 계산 (2단계 결과 활용)"""
        print("\n📐 향상된 변동계수 계산 중...")
        
        if self.lp_data is None:
            print("   ❌ LP 데이터가 로딩되지 않았습니다.")
            return {}
        
        # 2단계 결과에서 시간 패턴 정보 가져오기 (실제 분석된 값 활용)
        temporal_patterns = self.step2_results.get('temporal_patterns', {})
        peak_hours = temporal_patterns.get('peak_hours', [])
        off_peak_hours = temporal_patterns.get('off_peak_hours', [])
        weekend_ratio = temporal_patterns.get('weekend_ratio', 1.0)
        
        # 2단계 결과가 없는 경우에만 기본값 사용
        if not peak_hours:
            peak_hours = [9, 10, 11, 14, 15, 18, 19]
            print(f"   ⚠️ 2단계 피크 시간 없음, 기본값 사용")
        if not off_peak_hours:
            off_peak_hours = [0, 1, 2, 3, 4, 5]
            print(f"   ⚠️ 2단계 비피크 시간 없음, 기본값 사용")
        
        print(f"   🕐 피크 시간: {peak_hours} (2단계 분석 결과)")
        print(f"   🌙 비피크 시간: {off_peak_hours} (2단계 분석 결과)")
        print(f"   📅 주말/평일 비율: {weekend_ratio:.3f} (2단계 분석 결과)")
        
    def calculate_enhanced_volatility_coefficient(self, optimize_weights=True):
        """향상된 변동계수 계산 (2단계 결과 활용)"""
        print("\n📐 향상된 변동계수 계산 중...")
        
        if self.lp_data is None:
            print("   ❌ LP 데이터가 로딩되지 않았습니다.")
            return {}
        
        # 2단계 결과에서 시간 패턴 정보 가져오기 (실제 분석된 값 활용)
        temporal_patterns = self.step2_results.get('temporal_patterns', {})
        peak_hours = temporal_patterns.get('peak_hours', [])
        off_peak_hours = temporal_patterns.get('off_peak_hours', [])
        weekend_ratio = temporal_patterns.get('weekend_ratio', 1.0)
        
        # 2단계 결과가 없는 경우에만 기본값 사용
        if not peak_hours:
            peak_hours = [9, 10, 11, 14, 15, 18, 19]
            print(f"   ⚠️ 2단계 피크 시간 없음, 기본값 사용")
        if not off_peak_hours:
            off_peak_hours = [0, 1, 2, 3, 4, 5]
            print(f"   ⚠️ 2단계 비피크 시간 없음, 기본값 사용")
        
        print(f"   🕐 피크 시간: {peak_hours} (2단계 분석 결과)")
        print(f"   🌙 비피크 시간: {off_peak_hours} (2단계 분석 결과)")
        print(f"   📅 주말/평일 비율: {weekend_ratio:.3f} (2단계 분석 결과)")
        
        # 시간 파생 변수 생성
        self.lp_data['hour'] = self.lp_data['datetime'].dt.hour
        self.lp_data['weekday'] = self.lp_data['datetime'].dt.weekday
        self.lp_data['is_weekend'] = self.lp_data['weekday'].isin([5, 6])
        self.lp_data['month'] = self.lp_data['datetime'].dt.month
        
        customers = self.lp_data['대체고객번호'].unique()
        volatility_results = {}
        volatility_components = []  # 가중치 최적화용
        processed_count = 0
        
        print(f"   👥 분석 대상: {len(customers)}명")
        
        # 1차: 모든 성분 계산
        batch_size = 100
        for i in range(0, len(customers), batch_size):
            batch_customers = customers[i:i+batch_size]
            
            for customer_id in batch_customers:
                customer_data = self.lp_data[self.lp_data['대체고객번호'] == customer_id].copy()
                
                if len(customer_data) < 96:  # 최소 1일 데이터 필요
                    continue
                
                try:
                    power_values = customer_data['순방향 유효전력'].values
                    
                    # 1. 기본 변동계수
                    basic_cv = np.std(power_values) / np.mean(power_values) if np.mean(power_values) > 0 else 0
                    
                    # 2. 시간대별 변동계수 (2단계 피크 정보 활용)
                    hourly_avg = customer_data.groupby('hour')['순방향 유효전력'].mean()
                    hourly_cv = np.std(hourly_avg) / np.mean(hourly_avg) if np.mean(hourly_avg) > 0 else 0
                    
                    # 3. 피크/비피크 변동성 (가중 적용)
                    peak_data = customer_data[customer_data['hour'].isin(peak_hours)]['순방향 유효전력']
                    off_peak_data = customer_data[customer_data['hour'].isin(off_peak_hours)]['순방향 유효전력']
                    
                    peak_cv = np.std(peak_data) / np.mean(peak_data) if len(peak_data) > 0 and np.mean(peak_data) > 0 else 0
                    off_peak_cv = np.std(off_peak_data) / np.mean(off_peak_data) if len(off_peak_data) > 0 and np.mean(off_peak_data) > 0 else 0
                    
                    # 4. 주말/평일 변동성 (2단계 비율 활용)
                    weekday_data = customer_data[~customer_data['is_weekend']]['순방향 유효전력']
                    weekend_data = customer_data[customer_data['is_weekend']]['순방향 유효전력']
                    
                    weekday_cv = np.std(weekday_data) / np.mean(weekday_data) if len(weekday_data) > 0 and np.mean(weekday_data) > 0 else 0
                    weekend_cv = np.std(weekend_data) / np.mean(weekend_data) if len(weekend_data) > 0 and np.mean(weekend_data) > 0 else 0
                    weekend_diff = abs(weekday_cv - weekend_cv) * weekend_ratio
                    
                    # 5. 계절별 변동성 (월별)
                    monthly_avg = customer_data.groupby('month')['순방향 유효전력'].mean()
                    seasonal_cv = np.std(monthly_avg) / np.mean(monthly_avg) if len(monthly_avg) > 1 and np.mean(monthly_avg) > 0 else 0
                    
                    # 부가 지표들
                    mean_power = np.mean(power_values)
                    max_power = np.max(power_values)
                    load_factor = mean_power / max_power if max_power > 0 else 0
                    
                    # 이상 패턴 지표
                    zero_ratio = (power_values == 0).sum() / len(power_values)
                    sudden_changes = pd.Series(power_values).pct_change().abs()
                    extreme_changes = (sudden_changes > 1.5).sum()
                    
                    # 피크/비피크 부하 비율
                    peak_avg = np.mean(peak_data) if len(peak_data) > 0 else 0
                    off_peak_avg = np.mean(off_peak_data) if len(off_peak_data) > 0 else 0
                    peak_load_ratio = peak_avg / off_peak_avg if off_peak_avg > 0 else 1.0
                    
                    # 가중치 최적화용 데이터 저장
                    volatility_components.append({
                        'customer_id': customer_id,
                        'basic_cv': basic_cv,
                        'hourly_cv': hourly_cv,
                        'peak_cv': peak_cv,
                        'weekend_diff': weekend_diff,
                        'seasonal_cv': seasonal_cv,
                        'load_factor': load_factor,
                        'zero_ratio': zero_ratio,
                        'extreme_changes': extreme_changes,
                        'data_points': len(power_values)
                    })
                    
                    processed_count += 1
                    
                except Exception as e:
                    print(f"   ⚠️ 고객 {customer_id} 계산 실패: {e}")
                    continue
            
            # 진행상황 출력
            if (i // batch_size + 1) % 10 == 0:
                print(f"   📊 진행: {min(i + batch_size, len(customers))}/{len(customers)} ({processed_count}명 완료)")
        
        # 2차: 가중치 최적화 (옵션)
        if optimize_weights and len(volatility_components) >= 20:
            optimal_weights = self.optimize_volatility_weights(volatility_components)
        else:
            optimal_weights = [0.35, 0.25, 0.20, 0.10, 0.10]  # 기본 가중치
            if optimize_weights:
                print(f"   ⚠️ 데이터 부족으로 기본 가중치 사용: {optimal_weights}")
        
        print(f"   🎯 최종 가중치: {[round(w, 3) for w in optimal_weights]}")
        
        # 3차: 최적 가중치로 최종 변동계수 계산
        for component in volatility_components:
            customer_id = component['customer_id']
            
            # 최적화된 가중치 적용
            enhanced_volatility_coefficient = (
                optimal_weights[0] * component['basic_cv'] +
                optimal_weights[1] * component['hourly_cv'] +
                optimal_weights[2] * component['peak_cv'] +
                optimal_weights[3] * component['weekend_diff'] +
                optimal_weights[4] * component['seasonal_cv']
            )
            
            volatility_results[customer_id] = {
                # 핵심 변동계수
                'enhanced_volatility_coefficient': round(enhanced_volatility_coefficient, 4),
                
                # 세부 변동성 지표
                'basic_cv': round(component['basic_cv'], 4),
                'hourly_cv': round(component['hourly_cv'], 4),
                'peak_cv': round(component['peak_cv'], 4),
                'weekend_diff': round(component['weekend_diff'], 4),
                'seasonal_cv': round(component['seasonal_cv'], 4),
                
                # 사용 패턴 지표  
                'load_factor': round(component['load_factor'], 4),
                'zero_ratio': round(component['zero_ratio'], 4),
                'extreme_changes': int(component['extreme_changes']),
                'data_points': component['data_points'],
                
                # 최적화 정보
                'optimized_weights': [round(w, 3) for w in optimal_weights]
            }
        
        print(f"   ✅ {len(volatility_results)}명 변동계수 계산 완료")
        
        if len(volatility_results) > 0:
            cv_values = [v['enhanced_volatility_coefficient'] for v in volatility_results.values()]
            print(f"   📈 평균 변동계수: {np.mean(cv_values):.4f}")
            print(f"   📊 변동계수 범위: {np.min(cv_values):.4f} ~ {np.max(cv_values):.4f}")
        
        return volatility_results
    
    def train_stacking_ensemble_model(self, volatility_results):
        """스태킹 앙상블 모델 훈련 (영업활동 변화 예측)"""
        print("\n🎯 스태킹 앙상블 모델 훈련 중...")
        
        if len(volatility_results) < 20:
            print("   ❌ 훈련 데이터가 부족합니다 (최소 20개 필요)")
            return None
        
        # 특성 준비
        features = []
        targets = []
        customer_ids = []
        
        for customer_id, data in volatility_results.items():
            feature_vector = [
                data['basic_cv'],
                data['hourly_cv'],
                data['peak_cv'],
                data['off_peak_cv'],
                data['weekday_cv'],
                data['weekend_cv'],
                data['seasonal_cv'],
                data['load_factor'],
                data['peak_load_ratio'],
                data['mean_power'],
                data['zero_ratio'],
                data['extreme_changes'] / data['data_points']  # 정규화된 극값 변화 비율
            ]
            features.append(feature_vector)
            targets.append(data['enhanced_volatility_coefficient'])
            customer_ids.append(customer_id)
        
        X = np.array(features)
        y = np.array(targets)
        
        print(f"   📊 훈련 데이터: {len(X)}개 샘플, {X.shape[1]}개 특성")
        
        # 데이터 분할
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 정규화
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Level-0 모델들 (다양성 확보하되 간결하게)
        self.level0_models = {
            'rf': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
            'gbm': GradientBoostingRegressor(n_estimators=100, max_depth=6, random_state=42),
            'ridge': Ridge(alpha=1.0)
        }
        
        # Level-0 예측값 생성 (3-Fold CV)
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        
        meta_features_train = np.zeros((len(X_train_scaled), len(self.level0_models)))
        meta_features_test = np.zeros((len(X_test_scaled), len(self.level0_models)))
        
        print(f"   🔄 Level-0 모델 훈련:")
        for i, (name, model) in enumerate(self.level0_models.items()):
            # 훈련 세트에 대한 CV 예측
            fold_predictions = np.zeros(len(X_train_scaled))
            for train_idx, val_idx in kf.split(X_train_scaled):
                fold_model = type(model)(**model.get_params())
                fold_model.fit(X_train_scaled[train_idx], y_train[train_idx])
                fold_predictions[val_idx] = fold_model.predict(X_train_scaled[val_idx])
            
            meta_features_train[:, i] = fold_predictions
            
            # 전체 훈련 세트로 재훈련 후 테스트 예측
            model.fit(X_train_scaled, y_train)
            meta_features_test[:, i] = model.predict(X_test_scaled)
            
            # 개별 모델 성능
            cv_pred = model.predict(X_test_scaled)
            cv_mae = mean_absolute_error(y_test, cv_pred)
            cv_r2 = r2_score(y_test, cv_pred)
            print(f"      {name}: MAE={cv_mae:.4f}, R²={cv_r2:.4f}")
        
        # Level-1 메타 모델 (Linear Regression)
        self.meta_model = LinearRegression()
        self.meta_model.fit(meta_features_train, y_train)
        
        # 최종 예측 및 성능 평가
        final_pred = self.meta_model.predict(meta_features_test)
        final_mae = mean_absolute_error(y_test, final_pred)
        final_r2 = r2_score(y_test, final_pred)
        
        print(f"   ✅ 스태킹 앙상블 훈련 완료")
        print(f"      최종 MAE: {final_mae:.4f}")
        print(f"      최종 R²: {final_r2:.4f}")
        
        # 특성 중요도 (Random Forest 기준)
        feature_names = [
            'basic_cv', 'hourly_cv', 'peak_cv', 'off_peak_cv',
            'weekday_cv', 'weekend_cv', 'seasonal_cv', 'load_factor',
            'peak_load_ratio', 'mean_power', 'zero_ratio', 'extreme_change_ratio'
        ]
        
        rf_importance = self.level0_models['rf'].feature_importances_
        print(f"   📊 주요 특성 중요도 (상위 5개):")
        importance_pairs = list(zip(feature_names, rf_importance))
        importance_pairs.sort(key=lambda x: x[1], reverse=True)
        for name, importance in importance_pairs[:5]:
            print(f"      {name}: {importance:.4f}")
        
        return {
            'final_mae': final_mae,
            'final_r2': final_r2,
            'level0_models': list(self.level0_models.keys()),
            'meta_model': 'LinearRegression',
            'feature_importance': dict(importance_pairs),
            'n_samples': len(X),
            'n_features': X.shape[1]
        }
    
    def analyze_business_stability(self, volatility_results):
        """영업활동 안정성 분석"""
        print("\n🔍 영업활동 안정성 분석 중...")
        
        coefficients = [v['enhanced_volatility_coefficient'] for v in volatility_results.values()]
        
        # 안정성 기준 설정 (분위수 기반)
        p10, p25, p50, p75, p90 = np.percentile(coefficients, [10, 25, 50, 75, 90])
        
        print(f"   📊 변동계수 분위수:")
        print(f"      10%: {p10:.4f}")
        print(f"      25%: {p25:.4f}")
        print(f"      50%: {p50:.4f}")
        print(f"      75%: {p75:.4f}")
        print(f"      90%: {p90:.4f}")
        
        stability_analysis = {}
        grade_counts = {'매우안정': 0, '안정': 0, '보통': 0, '주의': 0, '불안정': 0}
        
        for customer_id, data in volatility_results.items():
            coeff = data['enhanced_volatility_coefficient']
            
            # 안정성 등급 분류 (5단계)
            if coeff <= p10:
                grade = '매우안정'
                risk_level = 'very_low'
            elif coeff <= p25:
                grade = '안정'
                risk_level = 'low'
            elif coeff <= p75:
                grade = '보통'
                risk_level = 'medium'
            elif coeff <= p90:
                grade = '주의'
                risk_level = 'high'
            else:
                grade = '불안정'
                risk_level = 'very_high'
            
            grade_counts[grade] += 1
            
            # 영업활동 변화 가능성 추정 (0~1)
            change_probability = min(0.95, max(0.05, (coeff - p25) / (p90 - p25))) if p90 > p25 else 0.5
            
            # 주요 위험 요인 식별
            risk_factors = []
            if data['peak_cv'] > data['basic_cv'] * 1.5:
                risk_factors.append('피크시간_불안정')
            if data['zero_ratio'] > 0.1:
                risk_factors.append('빈번한_사용중단')
            if data['extreme_changes'] > data['data_points'] * 0.05:
                risk_factors.append('급격한_변화')
            if data['peak_load_ratio'] > 3.0:
                risk_factors.append('피크부하_집중')
            
            stability_analysis[customer_id] = {
                'enhanced_volatility_coefficient': round(coeff, 4),
                'stability_grade': grade,
                'risk_level': risk_level,
                'change_probability': round(change_probability, 3),
                'risk_factors': risk_factors,
                'load_factor': data['load_factor'],
                'peak_load_ratio': data['peak_load_ratio']
            }
        
        # 등급별 분포 출력
        print(f"   📋 안정성 등급 분포:")
        total_customers = len(stability_analysis)
        for grade, count in grade_counts.items():
            percentage = count / total_customers * 100 if total_customers > 0 else 0
            print(f"      {grade}: {count}명 ({percentage:.1f}%)")
        
        return stability_analysis
    
    def generate_comprehensive_report(self, volatility_results, model_performance, stability_analysis):
        """종합 분석 리포트 생성"""
        print("\n📋 종합 분석 리포트 생성 중...")
        
        # 기본 통계
        coefficients = [v['enhanced_volatility_coefficient'] for v in volatility_results.values()]
        
        # 위험 고객 식별
        high_risk_customers = [
            customer_id for customer_id, analysis in stability_analysis.items()
            if analysis['risk_level'] in ['high', 'very_high']
        ]
        
        # 주요 위험 요인 집계
        all_risk_factors = []
        for analysis in stability_analysis.values():
            all_risk_factors.extend(analysis['risk_factors'])
        
        from collections import Counter
        risk_factor_counts = Counter(all_risk_factors)
        
        report = {
            'analysis_metadata': {
                'timestamp': pd.Timestamp.now().isoformat(),
                'step1_results_used': bool(self.step1_results),
                'step2_results_used': bool(self.step2_results),
                'kepco_data_used': self.kepco_data is not None,
                'total_customers_analyzed': len(volatility_results)
            },
            
            'volatility_coefficient_summary': {
                'total_customers': len(volatility_results),
                'mean_coefficient': round(np.mean(coefficients), 4),
                'std_coefficient': round(np.std(coefficients), 4),
                'percentiles': {
                    '10%': round(np.percentile(coefficients, 10), 4),
                    '25%': round(np.percentile(coefficients, 25), 4),
                    '50%': round(np.percentile(coefficients, 50), 4),
                    '75%': round(np.percentile(coefficients, 75), 4),
                    '90%': round(np.percentile(coefficients, 90), 4)
                }
            },
            
            'stacking_model_performance': model_performance,
            
            'business_stability_distribution': {
                grade: sum(1 for a in stability_analysis.values() if a['stability_grade'] == grade)
                for grade in ['매우안정', '안정', '보통', '주의', '불안정']
            },
            
            'risk_analysis': {
                'high_risk_customers': len(high_risk_customers),
                'high_risk_percentage': round(len(high_risk_customers) / len(stability_analysis) * 100, 1),
                'top_risk_factors': dict(risk_factor_counts.most_common(5))
            },
            
            'business_insights': [
                f"총 {len(volatility_results)}명 고객의 전력 사용패턴 변동계수 분석 완료",
                f"스태킹 앙상블 모델 예측 정확도(R²): {model_performance['final_r2']:.3f}",
                f"고위험 고객 {len(high_risk_customers)}명 식별 (전체의 {len(high_risk_customers)/len(stability_analysis)*100:.1f}%)",
                f"주요 위험 요인: {list(risk_factor_counts.keys())[:3]}",
                "실시간 모니터링 및 예방적 관리 체계 구축 가능"
            ],
            
            'recommendations': [
                "고위험 고객에 대한 집중 모니터링 체계 구축",
                "피크시간 전력 사용 패턴 최적화 지원",
                "예측 모델 기반 선제적 고객 관리",
                "업종별 맞춤형 전력 효율성 개선 프로그램 개발"
            ]
        }
        
        return report

# ===== 실행 예제 =====

def main():
    """메인 실행 함수"""
    print("🏆 한국전력공사 전력 사용패턴 변동계수 스태킹 분석 시스템")
    print("=" * 70)
    print("📁 기존 전처리 결과 활용 버전")
    print()
    
    try:
        # 1. 분석기 초기화
        print("1️⃣ 분석기 초기화")
        analyzer = KEPCOStackingVolatilityAnalyzer('./analysis_results')
        
        # 2. 전처리된 데이터 로딩
        print("\n2️⃣ 전처리된 데이터 로딩")
        if not analyzer.load_preprocessed_data():
            print("❌ 데이터 로딩 실패. 1-2단계 전처리를 먼저 실행하세요.")
            return None
        
        # 3. 향상된 변동계수 계산
        print("\n3️⃣ 향상된 변동계수 계산")
        volatility_results = analyzer.calculate_enhanced_volatility_coefficient()
        
        if not volatility_results:
            print("❌ 변동계수 계산 실패")
            return None
        
        # 4. 스태킹 앙상블 모델 훈련
        print("\n4️⃣ 스태킹 앙상블 모델 훈련")
        model_performance = analyzer.train_stacking_ensemble_model(volatility_results)
        
        if not model_performance:
            print("❌ 모델 훈련 실패")
            return None
        
        # 5. 영업활동 안정성 분석
        print("\n5️⃣ 영업활동 안정성 분석")
        stability_analysis = analyzer.analyze_business_stability(volatility_results)
        
        # 6. 종합 리포트 생성
        print("\n6️⃣ 종합 리포트 생성")
        comprehensive_report = analyzer.generate_comprehensive_report(
            volatility_results, model_performance, stability_analysis
        )
        
        # 7. 결과 저장
        print("\n7️⃣ 결과 저장")
        save_results(volatility_results, stability_analysis, comprehensive_report)
        
        # 8. 최종 요약 출력
        print_final_summary(comprehensive_report, model_performance)
        
        return {
            'volatility_results': volatility_results,
            'model_performance': model_performance,
            'stability_analysis': stability_analysis,
            'comprehensive_report': comprehensive_report
        }
        
    except Exception as e:
        print(f"❌ 시스템 실행 중 오류: {e}")
        import traceback
        traceback.print_exc()
        return None

def save_results(volatility_results, stability_analysis, comprehensive_report):
    """분석 결과 저장"""
    try:
        import json
        from datetime import datetime
        
        # 결과 디렉토리 생성
        os.makedirs('./analysis_results', exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 1. 변동계수 결과 저장 (CSV)
        volatility_df = pd.DataFrame.from_dict(volatility_results, orient='index')
        volatility_df.reset_index(inplace=True)
        volatility_df.rename(columns={'index': '대체고객번호'}, inplace=True)
        
        volatility_csv_path = f'./analysis_results/volatility_coefficients_{timestamp}.csv'
        volatility_df.to_csv(volatility_csv_path, index=False, encoding='utf-8-sig')
        print(f"   💾 변동계수 결과: {volatility_csv_path}")
        
        # 2. 안정성 분석 결과 저장 (CSV)
        stability_df = pd.DataFrame.from_dict(stability_analysis, orient='index')
        stability_df.reset_index(inplace=True)
        stability_df.rename(columns={'index': '대체고객번호'}, inplace=True)
        
        # 위험 요인을 문자열로 변환
        stability_df['risk_factors_str'] = stability_df['risk_factors'].apply(
            lambda x: ', '.join(x) if x else ''
        )
        
        stability_csv_path = f'./analysis_results/business_stability_{timestamp}.csv'
        stability_df.to_csv(stability_csv_path, index=False, encoding='utf-8-sig')
        print(f"   💾 안정성 분석: {stability_csv_path}")
        
        # 3. 종합 리포트 저장 (JSON)
        report_json_path = f'./analysis_results/comprehensive_report_{timestamp}.json'
        with open(report_json_path, 'w', encoding='utf-8') as f:
            json.dump(comprehensive_report, f, ensure_ascii=False, indent=2, default=str)
        print(f"   💾 종합 리포트: {report_json_path}")
        
        # 4. 3단계 결과 통합 저장 (다음 단계 연계용)
        final_results = {
            'metadata': {
                'stage': 'step3_stacking_volatility_analysis',
                'timestamp': datetime.now().isoformat(),
                'version': '3.0',
                'total_customers': len(volatility_results)
            },
            'volatility_summary': comprehensive_report['volatility_coefficient_summary'],
            'model_performance': comprehensive_report['stacking_model_performance'],
            'stability_distribution': comprehensive_report['business_stability_distribution'],
            'risk_analysis': comprehensive_report['risk_analysis'],
            'file_references': {
                'volatility_csv': volatility_csv_path,
                'stability_csv': stability_csv_path,
                'report_json': report_json_path
            }
        }
        
        final_json_path = './analysis_results/analysis_results3.json'
        with open(final_json_path, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, ensure_ascii=False, indent=2, default=str)
        print(f"   💾 3단계 통합 결과: {final_json_path}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ 결과 저장 실패: {e}")
        return False

def print_final_summary(comprehensive_report, model_performance):
    """최종 요약 출력"""
    print("\n" + "=" * 70)
    print("🎉 한국전력공사 전력 사용패턴 변동계수 스태킹 분석 완료!")
    print("=" * 70)
    
    # 핵심 성과
    print("📊 핵심 성과:")
    print(f"   ✅ 분석 고객: {comprehensive_report['volatility_coefficient_summary']['total_customers']:,}명")
    print(f"   ✅ 평균 변동계수: {comprehensive_report['volatility_coefficient_summary']['mean_coefficient']}")
    print(f"   ✅ 모델 예측 정확도(R²): {model_performance['final_r2']:.3f}")
    print(f"   ✅ 모델 오차(MAE): {model_performance['final_mae']:.4f}")
    
    # 안정성 분포
    print("\n🔍 고객 안정성 분포:")
    stability_dist = comprehensive_report['business_stability_distribution']
    total = sum(stability_dist.values())
    for grade, count in stability_dist.items():
        percentage = count / total * 100 if total > 0 else 0
        print(f"   {grade}: {count}명 ({percentage:.1f}%)")
    
    # 위험 분석
    risk_info = comprehensive_report['risk_analysis']
    print(f"\n⚠️ 위험 분석:")
    print(f"   고위험 고객: {risk_info['high_risk_customers']}명 ({risk_info['high_risk_percentage']}%)")
    print(f"   주요 위험 요인:")
    for factor, count in list(risk_info['top_risk_factors'].items())[:3]:
        print(f"      - {factor}: {count}건")
    
    # 기술적 성과
    print(f"\n🎯 기술적 성과:")
    print(f"   스태킹 앙상블 구성: {len(model_performance['level0_models'])}개 Level-0 모델")
    print(f"   특성 개수: {model_performance['n_features']}개")
    print(f"   훈련 샘플: {model_performance['n_samples']}개")
    
    # 비즈니스 가치
    print(f"\n💼 비즈니스 가치:")
    for insight in comprehensive_report['business_insights']:
        print(f"   • {insight}")
    
    # 권장사항
    print(f"\n📋 권장사항:")
    for recommendation in comprehensive_report['recommendations']:
        print(f"   • {recommendation}")
    
    print(f"\n🏆 공모전 제출 준비 완료!")
    print(f"   📁 결과 파일: ./analysis_results/ 디렉토리")
    print(f"   📊 핵심 알고리즘: 스태킹 앙상블 기반 변동계수")
    print(f"   🎯 실무 활용: 즉시 적용 가능한 고객 리스크 관리 시스템")

# 단독 실행용 테스트 데이터 생성 함수
def create_test_environment():
    """테스트 환경 생성 (실제 전처리 파일이 없을 때)"""
    print("🧪 테스트 환경 생성 중...")
    
    import json
    from datetime import datetime, timedelta
    
    # 테스트용 디렉토리 생성
    os.makedirs('./analysis_results', exist_ok=True)
    
    # 1. 1단계 결과 생성
    step1_results = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'analysis_stage': 'step1_preprocessing',
            'total_customers': 50,
            'total_lp_records': 48000
        },
        'customer_summary': {
            'total_customers': 50,
            'contract_types': {'222': 15, '226': 10, '322': 15, '726': 10},
            'usage_types': {'02': 20, '09': 30}
        },
        'lp_data_summary': {
            'total_records': 48000,
            'total_customers': 50,
            'avg_power': 75.5
        }
    }
    
    with open('./analysis_results/analysis_results.json', 'w', encoding='utf-8') as f:
        json.dump(step1_results, f, ensure_ascii=False, indent=2, default=str)
    
    # 2. 2단계 결과 생성 (실제 2단계 결과 형태로)
    step2_results = {
        'temporal_patterns': {
            'peak_hours': [10, 11, 14, 15, 18, 19],  # 실제 분석된 것처럼
            'off_peak_hours': [0, 1, 2, 3, 4, 5, 22, 23],
            'weekend_ratio': 0.72,
            'hourly_patterns': {
                'mean': {str(h): 50 + h*2 for h in range(24)}
            },
            'seasonal_patterns': {
                '봄': {'mean': 65.5}, '여름': {'mean': 85.2}, 
                '가을': {'mean': 70.1}, '겨울': {'mean': 90.3}
            }
        },
        'volatility_analysis': {
            'overall_cv': 0.35,
            'customer_cv_stats': {
                'mean': 0.32,
                'std': 0.15,
                'percentiles': {
                    '10%': 0.15, '25%': 0.22, '50%': 0.31, 
                    '75%': 0.41, '90%': 0.55
                }
            },
            'volatility_distribution': {
                '매우 안정 (<0.1)': 3, '안정 (0.1-0.2)': 8, 
                '보통 (0.2-0.3)': 15, '높음 (0.3-0.5)': 18, 
                '매우 높음 (0.5-1.0)': 5, '극히 높음 (>1.0)': 1
            }
        },
        'anomaly_analysis': {
            'processed_customers': 50,
            'anomaly_customers': {
                'high_night_usage': 3,
                'excessive_zeros': 2,
                'high_volatility': 6,
                'statistical_outliers': 4
            }
        }
    }
    
    with open('./analysis_results/analysis_results2.json', 'w', encoding='utf-8') as f:
        json.dump(step2_results, f, ensure_ascii=False, indent=2, default=str)
    
    # 3. 테스트용 LP 데이터 생성
    print("   📊 테스트 LP 데이터 생성 중...")
    
    np.random.seed(42)
    test_data = []
    
    # 50명 고객, 20일간, 15분 간격
    for customer in range(1, 51):
        for day in range(20):
            for hour in range(24):
                for minute in [0, 15, 30, 45]:
                    timestamp = datetime(2024, 3, 1) + timedelta(days=day, hours=hour, minutes=minute)
                    
                    # 고객별 다른 변동성 패턴
                    base_power = 50 + customer * 2
                    noise_level = 0.1 + (customer % 5) * 0.1
                    
                    # 시간대별 패턴 적용 (2단계 분석 결과 활용)
                    if hour in [10, 11, 14, 15, 18, 19]:  # 2단계에서 분석된 피크시간
                        base_power *= 1.4
                    elif hour in [0, 1, 2, 3, 4, 5, 22, 23]:  # 2단계에서 분석된 비피크시간
                        base_power *= 0.5
                    
                    power = base_power + np.random.normal(0, base_power * noise_level)
                    power = max(0, power)
                    
                    test_data.append({
                        '대체고객번호': f'TEST_{customer:03d}',
                        'datetime': timestamp,
                        '순방향 유효전력': round(power, 1),
                        '지상무효': round(power * 0.1, 1),
                        '진상무효': round(power * 0.05, 1),
                        '피상전력': round(power * 1.1, 1)
                    })
    
    test_df = pd.DataFrame(test_data)
    
    # HDF5로 저장 시도
    try:
        test_df.to_hdf('./analysis_results/processed_lp_data.h5', key='df', mode='w')
        print("   ✅ HDF5 테스트 데이터 생성 완료")
    except Exception as e:
        # CSV로 대체
        test_df.to_csv('./analysis_results/processed_lp_data.csv', index=False)
        print(f"   ✅ CSV 테스트 데이터 생성 완료 (HDF5 실패: {e})")
    
    print(f"   📊 생성된 데이터: {len(test_df):,}건")
    print(f"   👥 테스트 고객: {test_df['대체고객번호'].nunique()}명")
    print("   🎯 테스트 환경 준비 완료!")

if __name__ == "__main__":
    print("🚀 한국전력공사 전력 사용패턴 변동계수 스태킹 분석 시작!")
    
    # 실제 전처리 파일 존재 여부 확인
    required_files = [
        './analysis_results/analysis_results.json',
        './analysis_results/analysis_results2.json'
    ]
    
    data_files = [
        './analysis_results/processed_lp_data.h5',
        './analysis_results/processed_lp_data.csv'
    ]
    
    missing_required = [f for f in required_files if not os.path.exists(f)]
    missing_data = not any(os.path.exists(f) for f in data_files)
    
    if missing_required or missing_data:
        print("\n⚠️ 필수 전처리 파일이 없습니다.")
        print("   누락된 파일:")
        for f in missing_required:
            print(f"      - {f}")
        if missing_data:
            print(f"      - LP 데이터 파일 (HDF5 또는 CSV)")
        
        print("\n🧪 테스트 환경을 생성하시겠습니까? (y/n): ", end="")
        user_input = input().strip().lower()
        
        if user_input == 'y':
            create_test_environment()
            print("\n✅ 테스트 환경 생성 완료. 다시 실행합니다...\n")
        else:
            print("\n❌ 1-2단계 전처리를 먼저 실행하세요.")
            exit(1)
    
    # 메인 분석 실행
    results = main()
    
    if results:
        print(f"\n🎊 스태킹 앙상블 분석 성공적으로 완료!")
        print(f"   📁 결과 확인: ./analysis_results/ 디렉토리")
        print(f"   🏆 공모전 제출 준비 완료!")
    else:
        print(f"\n❌ 분석 실패")

# 스태킹 예측 함수 (추가 유틸리티)
def predict_new_customer(analyzer, customer_features):
    """새로운 고객의 변동성 예측"""
    if analyzer.level0_models and analyzer.meta_model:
        # Level-0 예측
        level0_preds = []
        scaled_features = analyzer.scaler.transform([customer_features])
        
        for model in analyzer.level0_models.values():
            pred = model.predict(scaled_features)[0]
            level0_preds.append(pred)
        
        # Level-1 예측
        meta_features = np.array([level0_preds])
        final_prediction = analyzer.meta_model.predict(meta_features)[0]
        
        return final_prediction
    else:
        raise ValueError("모델이 훈련되지 않았습니다.")

# 실시간 모니터링용 함수 (추가 유틸리티)
def monitor_customer_volatility(analyzer, customer_id, new_lp_data):
    """실시간 고객 변동성 모니터링"""
    # 새로운 LP 데이터로 변동계수 계산
    # ... (실제 구현은 calculate_enhanced_volatility_coefficient 로직 활용)
    pass