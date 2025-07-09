"""
한국전력공사 공모전: 창의적 전력 사용패턴 변동계수 2단계 알고리즘
'기업 경영활동 디지털 바이오마커 시스템'

🎯 핵심 창의 포인트:
1. "전력 DNA 분석" - 기업 고유의 전력 사용 지문 식별
2. "경영 건강도 진단" - 의학적 접근으로 기업 상태 평가  
3. "시간여행 예측" - 과거/현재/미래를 연결하는 변동성 예측
4. "업종별 AI 전문가" - 각 업종에 특화된 분석 엔진
5. "디지털 전환 감지" - 기업의 디지털화 수준 실시간 모니터링
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, IsolationForest
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats
from scipy.fft import fft, fftfreq
import warnings
warnings.filterwarnings('ignore')

class CreativePowerDNAAnalyzer:
    """
    🧬 창의적 전력 DNA 분석기
    기업의 전력 사용을 생체 신호처럼 분석하여 경영 상태 진단
    """
    
    def __init__(self, analysis_results_path='./analysis_result/analysis_results.json'):
        """
        초기화 - 1단계 전처리 결과 활용
        """
        self.analysis_results_path = analysis_results_path
        self.preprocessing_results = self._load_preprocessing_results()
        
        # 핵심 분석 엔진들
        self.dna_profiles = {}           # 전력 DNA 프로필
        self.health_diagnostics = {}     # 경영 건강도 진단
        self.time_travel_predictions = {} # 시간여행 예측
        self.industry_experts = {}       # 업종별 전문가
        self.digital_transformation = {} # 디지털 전환 지수
        
        print("🧬 창의적 전력 DNA 분석기 초기화")
        print("=" * 60)
        print("📋 분석 엔진:")
        print("  🔬 전력 DNA 시퀀싱")
        print("  🏥 경영 건강도 진단")
        print("  🕰️ 시간여행 변동성 예측")
        print("  👨‍💼 업종별 AI 전문가")
        print("  🚀 디지털 전환 감지기")
        print()
    
    def _load_preprocessing_results(self):
        """1단계 전처리 결과 로딩"""
        try:
            if os.path.exists(self.analysis_results_path):
                with open(self.analysis_results_path, 'r', encoding='utf-8') as f:
                    results = json.load(f)
                print(f"✅ 전처리 결과 로딩 성공: {self.analysis_results_path}")
                return results
            else:
                print(f"⚠️ 전처리 결과 파일 없음: {self.analysis_results_path}")
                return {}
        except Exception as e:
            print(f"❌ 전처리 결과 로딩 실패: {e}")
            return {}
    
    # ============ 1. 전력 DNA 시퀀싱 ============
    
    def extract_power_dna(self, lp_data):
        """
        🧬 전력 DNA 추출 - 기업의 고유한 전력 사용 지문 분석
        """
        print("🧬 전력 DNA 시퀀싱 중...")
        
        dna_profiles = {}
        customers = lp_data['대체고객번호'].unique()
        
        for customer in customers:
            customer_data = lp_data[lp_data['대체고객번호'] == customer].copy()
            customer_data['datetime'] = pd.to_datetime(customer_data['LP수신일자'])
            customer_data = customer_data.sort_values('datetime')
            
            # DNA 염기서열 구성 요소
            dna_sequence = {
                'A_gene': self._extract_activity_gene(customer_data),      # 활동성 유전자
                'T_gene': self._extract_timing_gene(customer_data),        # 시간성 유전자  
                'G_gene': self._extract_growth_gene(customer_data),        # 성장성 유전자
                'C_gene': self._extract_consistency_gene(customer_data)     # 일관성 유전자
            }
            
            # DNA 지문 생성
            dna_fingerprint = self._create_dna_fingerprint(dna_sequence)
            
            # 기업 DNA 타입 분류
            dna_type = self._classify_dna_type(dna_sequence)
            
            dna_profiles[customer] = {
                'dna_sequence': dna_sequence,
                'dna_fingerprint': dna_fingerprint,
                'dna_type': dna_type,
                'uniqueness_score': self._calculate_uniqueness_score(dna_sequence)
            }
        
        self.dna_profiles = dna_profiles
        print(f"✅ {len(customers)}개 기업 DNA 분석 완료")
        return dna_profiles
    
    def _extract_activity_gene(self, data):
        """활동성 유전자 - 전력 사용의 활발함 정도"""
        power_values = data['순방향유효전력'].values
        
        # 활동 강도 (평균 대비 피크 사용량)
        activity_intensity = np.max(power_values) / np.mean(power_values) if np.mean(power_values) > 0 else 0
        
        # 활동 빈도 (임계값 이상 사용 횟수)
        threshold = np.percentile(power_values, 75)
        activity_frequency = np.sum(power_values > threshold) / len(power_values)
        
        # 활동 지속성 (연속적인 고사용량 구간)
        high_usage_periods = self._find_continuous_periods(power_values > threshold)
        activity_persistence = np.mean([len(period) for period in high_usage_periods]) if high_usage_periods else 0
        
        return {
            'intensity': round(activity_intensity, 4),
            'frequency': round(activity_frequency, 4),
            'persistence': round(activity_persistence, 4)
        }
    
    def _extract_timing_gene(self, data):
        """시간성 유전자 - 시간 패턴의 규칙성"""
        data['hour'] = data['datetime'].dt.hour
        data['weekday'] = data['datetime'].dt.weekday
        
        # 시간 규칙성 (시간대별 사용 패턴의 일관성)
        hourly_pattern = data.groupby('hour')['순방향유효전력'].mean()
        timing_regularity = 1 / (hourly_pattern.std() / hourly_pattern.mean() + 1) if hourly_pattern.mean() > 0 else 0
        
        # 주기성 강도 (FFT를 통한 주파수 분석)
        power_series = data['순방향유효전력'].values
        fft_values = fft(power_series)
        frequencies = fftfreq(len(power_series))
        periodicity_strength = np.max(np.abs(fft_values[1:len(fft_values)//2]))
        
        # 업무시간 집중도 (9-18시 사용량 비중)
        business_hours = data[(data['hour'] >= 9) & (data['hour'] <= 18)]
        if len(data) > 0:
            business_concentration = business_hours['순방향유효전력'].sum() / data['순방향유효전력'].sum()
        else:
            business_concentration = 0
        
        return {
            'regularity': round(timing_regularity, 4),
            'periodicity': round(float(periodicity_strength), 4),
            'business_focus': round(business_concentration, 4)
        }
    
    def _extract_growth_gene(self, data):
        """성장성 유전자 - 사용량 변화 트렌드"""
        # 시간 순서대로 정렬
        data = data.sort_values('datetime')
        power_values = data['순방향유효전력'].values
        
        # 성장 트렌드 (선형 회귀 기울기)
        x = np.arange(len(power_values))
        if len(power_values) > 1:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, power_values)
            growth_trend = slope
            trend_confidence = abs(r_value)
        else:
            growth_trend = 0
            trend_confidence = 0
        
        # 성장 가속도 (2차 미분)
        if len(power_values) > 2:
            growth_acceleration = np.mean(np.diff(power_values, 2))
        else:
            growth_acceleration = 0
        
        # 성장 안정성 (트렌드의 일관성)
        window_size = min(96, len(power_values) // 4)  # 1일 또는 전체의 1/4
        if window_size > 1:
            rolling_trends = []
            for i in range(0, len(power_values) - window_size, window_size):
                window_data = power_values[i:i+window_size]
                window_x = np.arange(len(window_data))
                window_slope, _, _, _, _ = stats.linregress(window_x, window_data)
                rolling_trends.append(window_slope)
            
            growth_stability = 1 / (np.std(rolling_trends) + 1) if rolling_trends else 0
        else:
            growth_stability = 0
        
        return {
            'trend': round(growth_trend, 4),
            'acceleration': round(growth_acceleration, 4),
            'stability': round(growth_stability, 4),
            'confidence': round(trend_confidence, 4)
        }
    
    def _extract_consistency_gene(self, data):
        """일관성 유전자 - 사용 패턴의 예측가능성"""
        power_values = data['순방향유효전력'].values
        
        # 기본 일관성 (변동계수의 역수)
        if np.mean(power_values) > 0:
            basic_consistency = 1 / (np.std(power_values) / np.mean(power_values) + 1)
        else:
            basic_consistency = 0
        
        # 패턴 일관성 (자기상관)
        if len(power_values) > 1:
            autocorr = np.corrcoef(power_values[:-1], power_values[1:])[0, 1]
            pattern_consistency = abs(autocorr) if not np.isnan(autocorr) else 0
        else:
            pattern_consistency = 0
        
        # 예측가능성 (단순 예측 모델의 정확도)
        if len(power_values) > 10:
            # 이동평균 예측
            window = min(5, len(power_values) // 2)
            predictions = []
            actuals = []
            
            for i in range(window, len(power_values)):
                prediction = np.mean(power_values[i-window:i])
                actual = power_values[i]
                predictions.append(prediction)
                actuals.append(actual)
            
            if predictions:
                mape = np.mean(np.abs((np.array(actuals) - np.array(predictions)) / (np.array(actuals) + 1)))
                predictability = 1 / (mape + 1)
            else:
                predictability = 0
        else:
            predictability = 0
        
        return {
            'basic': round(basic_consistency, 4),
            'pattern': round(pattern_consistency, 4),
            'predictability': round(predictability, 4)
        }
    
    def _find_continuous_periods(self, boolean_array):
        """연속된 True 구간 찾기"""
        periods = []
        start = None
        
        for i, value in enumerate(boolean_array):
            if value and start is None:
                start = i
            elif not value and start is not None:
                periods.append(list(range(start, i)))
                start = None
        
        # 마지막 구간 처리
        if start is not None:
            periods.append(list(range(start, len(boolean_array))))
        
        return periods
    
    def _create_dna_fingerprint(self, dna_sequence):
        """DNA 지문 생성 - 고유 식별자"""
        # 각 유전자의 주요 특성을 결합하여 고유 지문 생성
        fingerprint_components = [
            dna_sequence['A_gene']['intensity'],
            dna_sequence['T_gene']['regularity'],
            dna_sequence['G_gene']['trend'],
            dna_sequence['C_gene']['basic']
        ]
        
        # 정규화 후 해시값 생성
        normalized = MinMaxScaler().fit_transform(np.array(fingerprint_components).reshape(-1, 1)).flatten()
        fingerprint = ''.join([f"{x:.2f}" for x in normalized])
        
        return fingerprint
    
    def _classify_dna_type(self, dna_sequence):
        """DNA 타입 분류"""
        activity_score = np.mean(list(dna_sequence['A_gene'].values()))
        timing_score = np.mean(list(dna_sequence['T_gene'].values()))
        growth_score = dna_sequence['G_gene']['trend']
        consistency_score = np.mean(list(dna_sequence['C_gene'].values()))
        
        # 창의적 DNA 타입 분류
        if activity_score > 0.7 and growth_score > 0:
            return "혁신 성장형 (Innovation Growth)"
        elif consistency_score > 0.8 and timing_score > 0.7:
            return "안정 운영형 (Stable Operation)"
        elif activity_score > 0.6 and timing_score < 0.4:
            return "유연 적응형 (Flexible Adaptation)"
        elif growth_score < -0.1:
            return "구조 조정형 (Restructuring)"
        else:
            return "균형 발전형 (Balanced Development)"
    
    def _calculate_uniqueness_score(self, dna_sequence):
        """DNA 고유성 점수"""
        # 각 유전자의 편차를 통해 고유성 측정
        all_values = []
        for gene in dna_sequence.values():
            all_values.extend(gene.values())
        
        uniqueness = np.std(all_values) * np.mean(all_values) if all_values else 0
        return round(uniqueness, 4)
    
    # ============ 2. 경영 건강도 진단 ============
    
    def diagnose_business_health(self, lp_data):
        """
        🏥 경영 건강도 진단 - 의학적 접근으로 기업 상태 평가
        """
        print("🏥 경영 건강도 진단 중...")
        
        health_diagnostics = {}
        customers = lp_data['대체고객번호'].unique()
        
        for customer in customers:
            customer_data = lp_data[lp_data['대체고객번호'] == customer].copy()
            
            # 종합 건강 검진
            vital_signs = self._check_vital_signs(customer_data)           # 생체 신호
            risk_factors = self._assess_risk_factors(customer_data)        # 위험 요소
            wellness_index = self._calculate_wellness_index(customer_data) # 웰니스 지수
            health_grade = self._assign_health_grade(vital_signs, risk_factors, wellness_index)
            
            health_diagnostics[customer] = {
                'vital_signs': vital_signs,
                'risk_factors': risk_factors,
                'wellness_index': wellness_index,
                'health_grade': health_grade,
                'diagnosis_date': datetime.now().isoformat()
            }
        
        self.health_diagnostics = health_diagnostics
        print(f"✅ {len(customers)}개 기업 건강 진단 완료")
        return health_diagnostics
    
    def _check_vital_signs(self, data):
        """경영 생체 신호 측정"""
        power_values = data['순방향유효전력'].values
        
        # 전력 맥박 (사용량의 주기적 변화)
        if len(power_values) > 1:
            power_pulse = np.mean(np.abs(np.diff(power_values)))
        else:
            power_pulse = 0
        
        # 전력 혈압 (최대/최소 사용량 비율)
        if np.min(power_values) > 0:
            power_pressure = np.max(power_values) / np.min(power_values)
        else:
            power_pressure = float('inf') if np.max(power_values) > 0 else 1
        
        # 전력 체온 (평균 사용 강도)
        power_temperature = np.mean(power_values)
        
        # 전력 호흡 (사용량 변동의 규칙성)
        if len(power_values) > 2:
            breath_intervals = np.diff(power_values)
            power_breathing = np.std(breath_intervals) / (np.mean(np.abs(breath_intervals)) + 1)
        else:
            power_breathing = 0
        
        return {
            'pulse': round(power_pulse, 2),
            'pressure': round(min(power_pressure, 999.99), 2),  # Cap at reasonable value
            'temperature': round(power_temperature, 2),
            'breathing': round(power_breathing, 4)
        }
    
    def _assess_risk_factors(self, data):
        """경영 위험 요소 평가"""
        power_values = data['순방향유효전력'].values
        
        # 급성 리스크 (급격한 변화)
        if len(power_values) > 1:
            sudden_changes = np.sum(np.abs(np.diff(power_values)) > 3 * np.std(power_values))
            acute_risk = sudden_changes / len(power_values)
        else:
            acute_risk = 0
        
        # 만성 리스크 (지속적인 불안정성)
        chronic_risk = np.std(power_values) / (np.mean(power_values) + 1)
        
        # 구조적 리스크 (비정상적 패턴)
        # 야간 과다 사용
        if 'datetime' not in data.columns:
            data['datetime'] = pd.to_datetime(data['LP수신일자'])
        data['hour'] = data['datetime'].dt.hour
        
        night_usage = data[(data['hour'] >= 22) | (data['hour'] <= 6)]['순방향유효전력'].mean()
        day_usage = data[(data['hour'] >= 9) & (data['hour'] <= 18)]['순방향유효전력'].mean()
        
        if day_usage > 0:
            structural_risk = night_usage / day_usage
        else:
            structural_risk = 0
        
        return {
            'acute': round(acute_risk, 4),
            'chronic': round(chronic_risk, 4),
            'structural': round(min(structural_risk, 5.0), 4)  # Cap at 5.0
        }
    
    def _calculate_wellness_index(self, data):
        """웰니스 지수 계산"""
        power_values = data['순방향유효전력'].values
        
        # 효율성 지수 (사용량 대비 안정성)
        if np.mean(power_values) > 0:
            efficiency_index = 1 / (np.std(power_values) / np.mean(power_values) + 1)
        else:
            efficiency_index = 0
        
        # 적응성 지수 (환경 변화 대응력)
        if len(power_values) > 10:
            # 시간에 따른 적응 능력
            first_half = power_values[:len(power_values)//2]
            second_half = power_values[len(power_values)//2:]
            
            adaptation_ability = 1 - abs(np.mean(first_half) - np.mean(second_half)) / (np.mean(power_values) + 1)
        else:
            adaptation_ability = 0.5
        
        # 지속성 지수 (장기간 운영 능력)
        sustainability_index = 1 - (np.sum(power_values == 0) / len(power_values))
        
        # 종합 웰니스 지수
        wellness_score = (efficiency_index * 0.4 + 
                         adaptation_ability * 0.3 + 
                         sustainability_index * 0.3)
        
        return {
            'efficiency': round(efficiency_index, 4),
            'adaptation': round(adaptation_ability, 4),
            'sustainability': round(sustainability_index, 4),
            'overall': round(wellness_score, 4)
        }
    
    def _assign_health_grade(self, vital_signs, risk_factors, wellness_index):
        """종합 건강 등급 산정"""
        # 생체 신호 정상성 평가
        vital_score = 1.0
        if vital_signs['pressure'] > 50:  # 과도한 변동성
            vital_score -= 0.3
        if vital_signs['breathing'] > 1.0:  # 불규칙한 패턴
            vital_score -= 0.2
        
        # 위험 요소 차감
        risk_penalty = (risk_factors['acute'] * 0.4 + 
                       risk_factors['chronic'] * 0.4 + 
                       min(risk_factors['structural'], 1.0) * 0.2)
        
        # 최종 건강 점수
        final_score = (vital_score * 0.3 + 
                      (1 - risk_penalty) * 0.3 + 
                      wellness_index['overall'] * 0.4)
        
        # 등급 산정
        if final_score >= 0.9:
            grade = "A+ (매우 건강)"
            status = "excellent"
        elif final_score >= 0.8:
            grade = "A (건강)"
            status = "good"
        elif final_score >= 0.7:
            grade = "B+ (양호)"
            status = "fair"
        elif final_score >= 0.6:
            grade = "B (보통)"
            status = "average"
        elif final_score >= 0.5:
            grade = "C+ (주의)"
            status = "caution"
        elif final_score >= 0.4:
            grade = "C (위험)"
            status = "risk"
        else:
            grade = "D (매우 위험)"
            status = "critical"
        
        return {
            'score': round(final_score, 4),
            'grade': grade,
            'status': status
        }
    
    # ============ 3. 시간여행 변동성 예측 ============
    
    def build_time_travel_predictor(self, lp_data):
        """
        🕰️ 시간여행 변동성 예측기 - 과거/현재/미래 연결
        """
        print("🕰️ 시간여행 예측기 구축 중...")
        
        predictions = {}
        customers = lp_data['대체고객번호'].unique()
        
        for customer in customers[:5]:  # 샘플로 5개 고객만
            customer_data = lp_data[lp_data['대체고객번호'] == customer].copy()
            
            # 시간여행 분석
            past_patterns = self._analyze_past_patterns(customer_data)
            present_state = self._assess_present_state(customer_data)
            future_forecast = self._predict_future_volatility(customer_data)
            
            predictions[customer] = {
                'past_patterns': past_patterns,
                'present_state': present_state,
                'future_forecast': future_forecast,
                'time_continuity_score': self._calculate_time_continuity(past_patterns, present_state, future_forecast)
            }
        
        self.time_travel_predictions = predictions
        print(f"✅ {len(predictions)}개 기업 시간여행 예측 완료")
        return predictions
    
    def _analyze_past_patterns(self, data):
        """과거 패턴 분석"""
        # 과거 데이터를 3분할하여 트렌드 분석
        data_sorted = data.sort_values('LP수신일자')
        power_values = data_sorted['순방향유효전력'].values
        
        if len(power_values) < 3:
            return {'trend': 0, 'stability': 0, 'cycles': 0}
        
        # 3개 구간으로 분할
        segment_size = len(power_values) // 3
        past_segment = power_values[:segment_size]
        middle_segment = power_values[segment_size:2*segment_size]
        recent_segment = power_values[2*segment_size:]
        
        # 과거 트렌드
        past_trend = (np.mean(middle_segment) - np.mean(past_segment)) / (np.mean(past_segment) + 1)
        
        # 과거 안정성
        past_stability = 1 / (np.std(past_segment) / (np.mean(past_segment) + 1) + 1)
        
        # 주기성 발견
        cycles_detected = len(self._detect_cycles(power_values))
        
        return {
            'trend': round(past_trend, 4),
            'stability': round(past_stability, 4),
            'cycles': cycles_detected
        }
    
    def _assess_present_state(self, data):
        """현재 상태 평가"""
        # 최근 데이터로 현재 상태 분석
        recent_data = data.tail(min(96, len(data)))  # 최근 1일 또는 전체
        power_values = recent_data['순방향유효전력'].values
        
        if len(power_values) == 0:
            return {'volatility': 0, 'trend': 0, 'anomaly_score': 0}
        
        # 현재 변동성
        current_volatility = np.std(power_values) / (np.mean(power_values) + 1)
        
        # 현재 트렌드
        if len(power_values) > 1:
            x = np.arange(len(power_values))
            slope, _, _, _, _ = stats.linregress(x, power_values)
            current_trend = slope
        else:
            current_trend = 0
        
        # 이상치 점수
        if len(power_values) > 1:
            isolation_forest = IsolationForest(contamination=0.1, random_state=42)
            anomaly_scores = isolation_forest.fit_predict(power_values.reshape(-1, 1))
            anomaly_score = np.mean(anomaly_scores == -1)
        else:
            anomaly_score = 0
        
        return {
            'volatility': round(current_volatility, 4),
            'trend': round(current_trend, 4),
            'anomaly_score': round(anomaly_score, 4)
        }
    
    def _predict_future_volatility(self, data):
        """미래 변동성 예측"""
        power_values = data['순방향유효전력'].values
        
        if len(power_values) < 10:
            return {'volatility_forecast': 0, 'trend_forecast': 0, 'confidence': 0}
        
        # 간단한 시계열 예측 (이동평균 + 트렌드)
        window_size = min(10, len(power_values) // 2)
        
        # 최근 데이터로 트렌드 계산
        recent_values = power_values[-window_size:]
        x = np.arange(len(recent_values))
        slope, intercept, r_value, _, _ = stats.linregress(x, recent_values)
        
        # 미래 변동성 예측 (최근 변동성에 트렌드 적용)
        recent_volatility = np.std(recent_values) / (np.mean(recent_values) + 1)
        volatility_trend = slope / (np.mean(recent_values) + 1)
        future_volatility = recent_volatility + volatility_trend
        
        # 신뢰도 (상관계수 기반)
        confidence = abs(r_value)
        
        return {
            'volatility_forecast': round(max(0, future_volatility), 4),
            'trend_forecast': round(slope, 4),
            'confidence': round(confidence, 4)
        }
    
    def _detect_cycles(self, power_values):
        """주기성 탐지"""
        if len(power_values) < 20:
            return []
        
        # FFT를 사용한 주기 탐지
        fft_values = fft(power_values)
        frequencies = fftfreq(len(power_values))
        
        # 주요 주파수 성분 찾기
        magnitude = np.abs(fft_values)
        peak_indices = np.where(magnitude > np.mean(magnitude) + 2 * np.std(magnitude))[0]
        
        cycles = []
        for idx in peak_indices[:5]:  # 상위 5개만
            if frequencies[idx] > 0:
                period = 1 / frequencies[idx]
                cycles.append(period)
        
        return cycles
    
    def _calculate_time_continuity(self, past, present, future):
        """시간 연속성 점수"""
        # 과거-현재-미래의 연결성 평가
        past_present_continuity = 1 - abs(past['trend'] - present['trend'])
        present_future_continuity = 1 - abs(present['trend'] - future['trend_forecast'])
        
        overall_continuity = (past_present_continuity + present_future_continuity) / 2
        return round(max(0, overall_continuity), 4)
    
    # ============ 4. 업종별 AI 전문가 시스템 ============
    
    def create_industry_experts(self, customer_data, lp_data):
        """
        👨‍💼 업종별 AI 전문가 시스템 구축
        """
        print("👨‍💼 업종별 AI 전문가 시스템 구축 중...")
        
        # 업종별 전문가 정의
        experts = {
            'manufacturing_expert': self._create_manufacturing_expert(),
            'commercial_expert': self._create_commercial_expert(),
            'service_expert': self._create_service_expert(),
            'general_expert': self._create_general_expert()
        }
        
        # 각 전문가의 분석 결과
        expert_analyses = {}
        
        for expert_name, expert_config in experts.items():
            expert_analyses[expert_name] = self._run_expert_analysis(
                expert_config, customer_data, lp_data
            )
        
        self.industry_experts = expert_analyses
        print(f"✅ {len(experts)}개 전문가 시스템 구축 완료")
        return expert_analyses
    
    def _create_manufacturing_expert(self):
        """제조업 전문가"""
        return {
            'name': '제조업 전문가',
            'specialty': '생산라인 효율성 분석',
            'key_indicators': [
                'shift_pattern_consistency',    # 교대 패턴 일관성
                'production_efficiency',        # 생산 효율성
                'equipment_utilization',        # 설비 가동률
                'energy_intensity'              # 에너지 집약도
            ],
            'risk_factors': [
                'production_halt_signals',      # 생산 중단 신호
                'equipment_degradation',        # 설비 노후화
                'shift_irregularities'          # 교대 불규칙성
            ],
            'thresholds': {
                'high_risk_cv': 0.8,
                'low_efficiency': 0.3,
                'abnormal_night_ratio': 0.1
            }
        }
    
    def _create_commercial_expert(self):
        """상업시설 전문가"""
        return {
            'name': '상업시설 전문가',
            'specialty': '고객 유입 패턴 분석',
            'key_indicators': [
                'business_hour_efficiency',     # 영업시간 효율성
                'customer_flow_pattern',        # 고객 유입 패턴
                'seasonal_adaptation',          # 계절적 적응성
                'peak_management'               # 피크시간 관리
            ],
            'risk_factors': [
                'declining_customer_flow',      # 고객 유입 감소
                'inefficient_operations',       # 비효율적 운영
                'seasonal_vulnerability'        # 계절적 취약성
            ],
            'thresholds': {
                'high_risk_cv': 0.6,
                'low_efficiency': 0.4,
                'weekend_dependency': 0.7
            }
        }
    
    def _create_service_expert(self):
        """서비스업 전문가"""
        return {
            'name': '서비스업 전문가',
            'specialty': '서비스 운영 최적화',
            'key_indicators': [
                'service_consistency',          # 서비스 일관성
                'operational_flexibility',      # 운영 유연성
                'resource_optimization',        # 자원 최적화
                'digital_readiness'             # 디지털 준비도
            ],
            'risk_factors': [
                'service_disruption',           # 서비스 중단
                'resource_waste',               # 자원 낭비
                'digital_lag'                   # 디지털 지연
            ],
            'thresholds': {
                'high_risk_cv': 0.7,
                'low_efficiency': 0.35,
                'digital_threshold': 0.2
            }
        }
    
    def _create_general_expert(self):
        """일반 업종 전문가"""
        return {
            'name': '일반 업종 전문가',
            'specialty': '종합적 경영 분석',
            'key_indicators': [
                'overall_stability',            # 전반적 안정성
                'growth_sustainability',        # 성장 지속가능성
                'risk_management',              # 위험 관리
                'operational_excellence'        # 운영 우수성
            ],
            'risk_factors': [
                'general_instability',          # 일반적 불안정성
                'growth_stagnation',            # 성장 정체
                'operational_inefficiency'      # 운영 비효율성
            ],
            'thresholds': {
                'high_risk_cv': 0.75,
                'low_efficiency': 0.3,
                'stagnation_threshold': 0.05
            }
        }
    
    def _run_expert_analysis(self, expert_config, customer_data, lp_data):
        """전문가 분석 실행"""
        analysis_results = {
            'expert_name': expert_config['name'],
            'specialty': expert_config['specialty'],
            'analyzed_customers': 0,
            'recommendations': [],
            'risk_alerts': [],
            'insights': []
        }
        
        # 샘플 고객들에 대해 전문가 분석 수행
        sample_customers = lp_data['대체고객번호'].unique()[:3]
        
        for customer in sample_customers:
            customer_lp = lp_data[lp_data['대체고객번호'] == customer]
            
            # 전문가별 지표 계산
            indicators = self._calculate_expert_indicators(
                expert_config, customer_lp
            )
            
            # 위험 요소 평가
            risks = self._assess_expert_risks(
                expert_config, customer_lp, indicators
            )
            
            # 권장사항 생성
            recommendations = self._generate_expert_recommendations(
                expert_config, indicators, risks
            )
            
            if risks:
                analysis_results['risk_alerts'].extend(risks)
            if recommendations:
                analysis_results['recommendations'].extend(recommendations)
        
        analysis_results['analyzed_customers'] = len(sample_customers)
        
        # 전문가별 인사이트 생성
        analysis_results['insights'] = self._generate_expert_insights(expert_config)
        
        return analysis_results
    
    def _calculate_expert_indicators(self, expert_config, customer_data):
        """전문가별 지표 계산"""
        power_values = customer_data['순방향유효전력'].values
        indicators = {}
        
        # 기본 통계
        if len(power_values) > 0:
            mean_power = np.mean(power_values)
            std_power = np.std(power_values)
            cv = std_power / mean_power if mean_power > 0 else 0
        else:
            mean_power = std_power = cv = 0
        
        # 업종별 특화 지표 계산
        if 'shift_pattern_consistency' in expert_config['key_indicators']:
            # 제조업: 교대 패턴 일관성
            indicators['shift_pattern_consistency'] = self._calculate_shift_consistency(customer_data)
        
        if 'business_hour_efficiency' in expert_config['key_indicators']:
            # 상업: 영업시간 효율성
            indicators['business_hour_efficiency'] = self._calculate_business_hour_efficiency(customer_data)
        
        if 'service_consistency' in expert_config['key_indicators']:
            # 서비스: 서비스 일관성
            indicators['service_consistency'] = 1 / (cv + 1)
        
        if 'overall_stability' in expert_config['key_indicators']:
            # 일반: 전반적 안정성
            indicators['overall_stability'] = 1 / (cv + 1)
        
        return indicators
    
    def _calculate_shift_consistency(self, data):
        """교대 패턴 일관성 계산"""
        if 'datetime' not in data.columns:
            data['datetime'] = pd.to_datetime(data['LP수신일자'])
        data['hour'] = data['datetime'].dt.hour
        
        # 3교대 패턴 가정 (0-8, 8-16, 16-24)
        shift_1 = data[(data['hour'] >= 0) & (data['hour'] < 8)]['순방향유효전력'].mean()
        shift_2 = data[(data['hour'] >= 8) & (data['hour'] < 16)]['순방향유효전력'].mean()
        shift_3 = data[(data['hour'] >= 16) & (data['hour'] < 24)]['순방향유효전력'].mean()
        
        shifts = [shift_1, shift_2, shift_3]
        shifts = [s for s in shifts if not np.isnan(s)]
        
        if len(shifts) > 1:
            shift_consistency = 1 / (np.std(shifts) / np.mean(shifts) + 1)
        else:
            shift_consistency = 0
        
        return round(shift_consistency, 4)
    
    def _calculate_business_hour_efficiency(self, data):
        """영업시간 효율성 계산"""
        if 'datetime' not in data.columns:
            data['datetime'] = pd.to_datetime(data['LP수신일자'])
        data['hour'] = data['datetime'].dt.hour
        
        # 영업시간 (9-21시) vs 비영업시간
        business_hours = data[(data['hour'] >= 9) & (data['hour'] <= 21)]['순방향유효전력'].mean()
        non_business_hours = data[(data['hour'] < 9) | (data['hour'] > 21)]['순방향유효전력'].mean()
        
        if non_business_hours > 0:
            efficiency = business_hours / (business_hours + non_business_hours)
        else:
            efficiency = 1.0
        
        return round(efficiency, 4)
    
    def _assess_expert_risks(self, expert_config, customer_data, indicators):
        """전문가별 위험 요소 평가"""
        risks = []
        thresholds = expert_config['thresholds']
        
        # 기본 변동계수 체크
        power_values = customer_data['순방향유효전력'].values
        if len(power_values) > 0:
            cv = np.std(power_values) / (np.mean(power_values) + 1)
            if cv > thresholds.get('high_risk_cv', 0.7):
                risks.append(f"높은 변동성 감지 (CV: {cv:.3f})")
        
        # 전문가별 특화 위험 평가
        for indicator, value in indicators.items():
            if 'efficiency' in indicator and value < thresholds.get('low_efficiency', 0.3):
                risks.append(f"낮은 {indicator}: {value:.3f}")
        
        return risks
    
    def _generate_expert_recommendations(self, expert_config, indicators, risks):
        """전문가별 권장사항 생성"""
        recommendations = []
        
        if risks:
            if expert_config['name'] == '제조업 전문가':
                recommendations.extend([
                    "생산 스케줄 최적화 검토",
                    "설비 효율성 개선 방안 수립",
                    "예측 정비 시스템 도입 고려"
                ])
            elif expert_config['name'] == '상업시설 전문가':
                recommendations.extend([
                    "고객 유입 패턴 분석 강화",
                    "피크시간 운영 최적화",
                    "계절별 운영 전략 수립"
                ])
            elif expert_config['name'] == '서비스업 전문가':
                recommendations.extend([
                    "서비스 표준화 프로세스 구축",
                    "디지털 전환 로드맵 수립",
                    "고객 경험 개선 방안 검토"
                ])
            else:  # 일반 업종
                recommendations.extend([
                    "종합적 운영 효율성 진단",
                    "리스크 관리 체계 구축",
                    "데이터 기반 의사결정 시스템 도입"
                ])
        
        return recommendations
    
    def _generate_expert_insights(self, expert_config):
        """전문가별 인사이트 생성"""
        insights = [
            f"{expert_config['name']}의 {expert_config['specialty']} 관점에서 분석 완료",
            f"주요 지표: {', '.join(expert_config['key_indicators'][:3])}",
            f"위험 요소: {', '.join(expert_config['risk_factors'][:2])}"
        ]
        return insights
    
    # ============ 5. 디지털 전환 감지기 ============
    
    def detect_digital_transformation(self, lp_data):
        """
        🚀 디지털 전환 감지기 - 기업의 디지털화 수준 실시간 모니터링
        """
        print("🚀 디지털 전환 감지 중...")
        
        digital_scores = {}
        customers = lp_data['대체고객번호'].unique()
        
        for customer in customers[:5]:  # 샘플 5개 고객
            customer_data = lp_data[lp_data['대체고객번호'] == customer].copy()
            
            # 디지털 전환 지표들
            automation_level = self._measure_automation_level(customer_data)
            remote_work_readiness = self._assess_remote_work_readiness(customer_data)
            smart_energy_usage = self._evaluate_smart_energy_usage(customer_data)
            digital_resilience = self._calculate_digital_resilience(customer_data)
            
            # 종합 디지털 전환 점수
            digital_transformation_score = self._calculate_digital_transformation_score(
                automation_level, remote_work_readiness, smart_energy_usage, digital_resilience
            )
            
            digital_scores[customer] = {
                'automation_level': automation_level,
                'remote_work_readiness': remote_work_readiness,
                'smart_energy_usage': smart_energy_usage,
                'digital_resilience': digital_resilience,
                'digital_transformation_score': digital_transformation_score,
                'digital_maturity_level': self._classify_digital_maturity(digital_transformation_score)
            }
        
        self.digital_transformation = digital_scores
        print(f"✅ {len(digital_scores)}개 기업 디지털 전환 분석 완료")
        return digital_scores
    
    def _measure_automation_level(self, data):
        """자동화 수준 측정"""
        if 'datetime' not in data.columns:
            data['datetime'] = pd.to_datetime(data['LP수신일자'])
        data['hour'] = data['datetime'].dt.hour
        
        # 야간 자동화 운영 (무인 운영 수준)
        night_hours = data[(data['hour'] >= 22) | (data['hour'] <= 6)]
        day_hours = data[(data['hour'] >= 9) & (data['hour'] <= 18)]
        
        if len(day_hours) > 0 and day_hours['순방향유효전력'].mean() > 0:
            night_automation = night_hours['순방향유효전력'].mean() / day_hours['순방향유효전력'].mean()
        else:
            night_automation = 0
        
        # 운영 일관성 (자동화된 시스템의 특징)
        power_values = data['순방향유효전력'].values
        if len(power_values) > 1:
            operation_consistency = 1 / (np.std(power_values) / (np.mean(power_values) + 1) + 1)
        else:
            operation_consistency = 0
        
        automation_score = (night_automation * 0.6 + operation_consistency * 0.4)
        return round(min(automation_score, 1.0), 4)
    
    def _assess_remote_work_readiness(self, data):
        """원격근무 준비도 평가"""
        if 'datetime' not in data.columns:
            data['datetime'] = pd.to_datetime(data['LP수신일자'])
        data['weekday'] = data['datetime'].dt.weekday
        
        # 주말/휴일 활동 (원격 접근성)
        weekend_data = data[data['weekday'].isin([5, 6])]  # 토, 일
        weekday_data = data[~data['weekday'].isin([5, 6])]
        
        if len(weekday_data) > 0 and weekday_data['순방향유효전력'].mean() > 0:
            remote_accessibility = weekend_data['순방향유효전력'].mean() / weekday_data['순방향유효전력'].mean()
        else:
            remote_accessibility = 0
        
        # 시간 유연성 (다양한 시간대 활용)
        hourly_usage = data.groupby(data['datetime'].dt.hour)['순방향유효전력'].mean()
        time_flexibility = 1 - (hourly_usage.std() / (hourly_usage.mean() + 1))
        
        remote_readiness = (remote_accessibility * 0.4 + time_flexibility * 0.6)
        return round(min(remote_readiness, 1.0), 4)
    
    def _evaluate_smart_energy_usage(self, data):
        """스마트 에너지 사용 평가"""
        power_values = data['순방향유효전력'].values
        
        if len(power_values) < 10:
            return 0
        
        # 에너지 효율성 (최적화된 사용 패턴)
        # 피크 시간 회피 정도
        if 'datetime' not in data.columns:
            data['datetime'] = pd.to_datetime(data['LP수신일자'])
        data['hour'] = data['datetime'].dt.hour
        
        peak_hours = [14, 15, 16, 17, 18, 19]  # 일반적 피크 시간
        off_peak_hours = [1, 2, 3, 4, 5, 22, 23]  # 저부하 시간
        
        peak_usage = data[data['hour'].isin(peak_hours)]['순방향유효전력'].mean()
        off_peak_usage = data[data['hour'].isin(off_peak_hours)]['순방향유효전력'].mean()
        
        if peak_usage > 0:
            peak_avoidance = 1 - (peak_usage / (peak_usage + off_peak_usage))
        else:
            peak_avoidance = 0
        
        # 에너지 사용 최적화 (변동성 최소화)
        energy_optimization = 1 / (np.std(power_values) / (np.mean(power_values) + 1) + 1)
        
        smart_usage = (peak_avoidance * 0.5 + energy_optimization * 0.5)
        return round(smart_usage, 4)
    
    def _calculate_digital_resilience(self, data):
        """디지털 복원력 계산"""
        power_values = data['순방향유효전력'].values
        
        if len(power_values) < 5:
            return 0
        
        # 시스템 안정성 (급격한 변화에 대한 복원력)
        stability_score = 1 - (np.sum(np.abs(np.diff(power_values)) > 2 * np.std(power_values)) / len(power_values))
        
        # 연속성 (0값 최소화)
        continuity_score = 1 - (np.sum(power_values == 0) / len(power_values))
        
        # 적응성 (점진적 변화 수용 능력)
        if len(power_values) > 10:
            first_half = power_values[:len(power_values)//2]
            second_half = power_values[len(power_values)//2:]
            adaptability = 1 - abs(np.mean(second_half) - np.mean(first_half)) / (np.mean(power_values) + 1)
        else:
            adaptability = 0.5
        
        resilience = (stability_score * 0.4 + continuity_score * 0.3 + adaptability * 0.3)
        return round(resilience, 4)
    
    def _calculate_digital_transformation_score(self, automation, remote, smart, resilience):
        """종합 디지털 전환 점수"""
        # 가중 평균으로 종합 점수 계산
        weights = {
            'automation': 0.3,
            'remote': 0.2,
            'smart': 0.25,
            'resilience': 0.25
        }
        
        total_score = (automation * weights['automation'] + 
                      remote * weights['remote'] + 
                      smart * weights['smart'] + 
                      resilience * weights['resilience'])
        
        return round(total_score, 4)
    
    def _classify_digital_maturity(self, score):
        """디지털 성숙도 분류"""
        if score >= 0.8:
            return "Level 5: 디지털 네이티브 (Digital Native)"
        elif score >= 0.7:
            return "Level 4: 디지털 선도 (Digital Leader)"
        elif score >= 0.6:
            return "Level 3: 디지털 적응 (Digital Adaptor)"
        elif score >= 0.4:
            return "Level 2: 디지털 학습 (Digital Learner)"
        elif score >= 0.2:
            return "Level 1: 디지털 시작 (Digital Beginner)"
        else:
            return "Level 0: 디지털 준비 (Digital Ready)"
    
    # ============ 6. 창의적 종합 변동계수 계산 ============
    
    def calculate_creative_volatility_coefficient(self, lp_data):
        """
        🎯 창의적 종합 변동계수 계산 - 모든 분석 결과 통합
        """
        print("🎯 창의적 종합 변동계수 계산 중...")
        
        # 모든 분석 수행
        dna_profiles = self.extract_power_dna(lp_data)
        health_diagnostics = self.diagnose_business_health(lp_data)
        time_predictions = self.build_time_travel_predictor(lp_data)
        digital_scores = self.detect_digital_transformation(lp_data)
        
        # 종합 변동계수 계산
        creative_coefficients = {}
        
        for customer in dna_profiles.keys():
            if customer in health_diagnostics and customer in digital_scores:
                
                # 각 차원별 점수 추출
                dna_score = dna_profiles[customer]['uniqueness_score']
                health_score = health_diagnostics[customer]['health_grade']['score']
                digital_score = digital_scores[customer]['digital_transformation_score']
                
                time_score = 0.5  # 기본값
                if customer in time_predictions:
                    time_score = time_predictions[customer]['time_continuity_score']
                
                # 창의적 변동계수 공식
                creative_vc = self._compute_creative_volatility_formula(
                    dna_score, health_score, digital_score, time_score
                )
                
                # 경영 활동 변화 예측
                business_change_prediction = self._predict_business_activity_change(
                    dna_profiles[customer], health_diagnostics[customer], digital_scores[customer]
                )
                
                creative_coefficients[customer] = {
                    'creative_volatility_coefficient': creative_vc,
                    'dna_contribution': dna_score,
                    'health_contribution': health_score,
                    'digital_contribution': digital_score,
                    'time_contribution': time_score,
                    'business_change_prediction': business_change_prediction,
                    'risk_level': self._assess_integrated_risk_level(creative_vc),
                    'recommendations': self._generate_integrated_recommendations(creative_vc, business_change_prediction)
                }
        
        print(f"✅ {len(creative_coefficients)}개 기업 창의적 변동계수 계산 완료")
        return creative_coefficients
    
    def _compute_creative_volatility_formula(self, dna, health, digital, time):
        """창의적 변동계수 공식"""
        # 혁신적 가중치 설계
        weights = {
            'business_stability': 0.35,    # 경영 안정성 (health 기반)
            'innovation_capacity': 0.25,   # 혁신 역량 (digital 기반)
            'unique_identity': 0.25,       # 고유성 (dna 기반)
            'predictability': 0.15         # 예측가능성 (time 기반)
        }
        
        # 정규화 및 가중 합산
        stability_component = health * weights['business_stability']
        innovation_component = digital * weights['innovation_capacity']
        identity_component = min(dna, 1.0) * weights['unique_identity']
        predictability_component = time * weights['predictability']
        
        # 창의적 변동계수 = 1 - 종합안정성점수 (높을수록 불안정)
        creative_vc = 1 - (stability_component + innovation_component + 
                          identity_component + predictability_component)
        
        return round(max(0, creative_vc), 4)
    
    def _predict_business_activity_change(self, dna_profile, health_diagnostic, digital_score):
        """경영 활동 변화 예측"""
        # DNA 타입별 변화 패턴
        dna_type = dna_profile['dna_type']
        health_grade = health_diagnostic['health_grade']['status']
        digital_level = digital_score['digital_maturity_level']
        
        # 변화 예측 로직
        if "혁신 성장형" in dna_type and "excellent" in health_grade:
            prediction = "급속한 디지털 확장 예상"
        elif "안정 운영형" in dna_type and "good" in health_grade:
            prediction = "점진적 효율성 개선 예상"
        elif "유연 적응형" in dna_type:
            prediction = "시장 변화에 따른 운영 조정 예상"
        elif "구조 조정형" in dna_type:
            prediction = "대규모 사업 재편 가능성"
        elif "critical" in health_grade or "risk" in health_grade:
            prediction = "긴급 경영 개선 조치 필요"
        else:
            prediction = "현 상태 유지 또는 소폭 변화 예상"
        
        return prediction
    
    def _assess_integrated_risk_level(self, creative_vc):
        """통합 위험 수준 평가"""
        if creative_vc >= 0.8:
            return "매우 높음 (즉시 조치 필요)"
        elif creative_vc >= 0.6:
            return "높음 (주의 깊은 모니터링)"
        elif creative_vc >= 0.4:
            return "보통 (정기적 점검)"
        elif creative_vc >= 0.2:
            return "낮음 (안정적 운영)"
        else:
            return "매우 낮음 (최적 상태)"
    
    def _generate_integrated_recommendations(self, creative_vc, business_prediction):
        """통합 권장사항 생성"""
        recommendations = []
        
        if creative_vc >= 0.7:
            recommendations.extend([
                "긴급 경영진 회의 소집",
                "전력 사용 패턴 상세 분석",
                "비상 운영 계획 활성화",
                "외부 전문가 컨설팅 검토"
            ])
        elif creative_vc >= 0.5:
            recommendations.extend([
                "월간 모니터링 체계 구축",
                "운영 효율성 개선 방안 수립",
                "예측 분석 시스템 도입",
                "리스크 관리 프로세스 강화"
            ])
        elif creative_vc >= 0.3:
            recommendations.extend([
                "분기별 성과 리뷰",
                "데이터 기반 의사결정 확대",
                "디지털 전환 로드맵 점검",
                "에너지 효율성 최적화"
            ])
        else:
            recommendations.extend([
                "현재 운영 방식 유지",
                "모범 사례 벤치마킹",
                "혁신 기회 탐색",
                "지속가능성 강화"
            ])
        
        # 비즈니스 변화 예측에 따른 추가 권장사항
        if "디지털 확장" in business_prediction:
            recommendations.append("디지털 인프라 투자 계획 수립")
        elif "구조 조정" in business_prediction:
            recommendations.append("변화 관리 전략 수립")
        elif "긴급 개선" in business_prediction:
            recommendations.append("즉시 실행 가능한 개선 과제 도출")
        
        return recommendations
    
    # ============ 7. 창의적 결과 리포트 생성 ============
    
    def generate_creative_report(self, lp_data, output_path='./creative_volatility_report.json'):
        """
        📋 창의적 분석 결과 종합 리포트 생성
        """
        print("📋 창의적 분석 결과 리포트 생성 중...")
        
        # 전체 분석 실행
        creative_results = self.calculate_creative_volatility_coefficient(lp_data)
        
        # 종합 리포트 구성
        comprehensive_report = {
            'report_metadata': {
                'title': '한국전력공사 창의적 전력 사용패턴 변동계수 분석 리포트',
                'subtitle': '기업 경영활동 디지털 바이오마커 시스템',
                'generated_at': datetime.now().isoformat(),
                'analysis_version': '2.0_creative',
                'total_customers_analyzed': len(creative_results)
            },
            
            'executive_summary': {
                'key_findings': self._generate_key_findings(creative_results),
                'overall_risk_distribution': self._analyze_risk_distribution(creative_results),
                'digital_maturity_overview': self._summarize_digital_maturity(),
                'business_health_summary': self._summarize_business_health(),
                'strategic_recommendations': self._generate_strategic_recommendations(creative_results)
            },
            
            'detailed_analysis': {
                'power_dna_insights': self._summarize_dna_insights(),
                'health_diagnostic_results': self._summarize_health_results(),
                'time_travel_predictions': self._summarize_time_predictions(),
                'industry_expert_findings': self._summarize_expert_findings(),
                'digital_transformation_analysis': self._summarize_digital_analysis()
            },
            
            'customer_profiles': creative_results,
            
            'methodology': {
                'innovative_approach': [
                    "전력 DNA 시퀀싱: 기업 고유의 전력 사용 지문 분석",
                    "경영 건강도 진단: 의학적 접근으로 기업 상태 평가",
                    "시간여행 예측: 과거-현재-미래 연결 변동성 예측",
                    "업종별 AI 전문가: 특화된 분석 엔진 활용",
                    "디지털 전환 감지: 실시간 디지털화 수준 모니터링"
                ],
                'creative_volatility_formula': {
                    'components': {
                        'business_stability': 0.35,
                        'innovation_capacity': 0.25,
                        'unique_identity': 0.25,
                        'predictability': 0.15
                    },
                    'interpretation': "1 - 종합안정성점수 (높을수록 변동성 큼)"
                }
            },
            
            'actionable_insights': {
                'immediate_actions': self._extract_immediate_actions(creative_results),
                'medium_term_strategy': self._extract_medium_term_strategy(creative_results),
                'long_term_vision': self._extract_long_term_vision(creative_results)
            }
        }
        
        # JSON 파일로 저장
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(comprehensive_report, f, ensure_ascii=False, indent=2, default=str)
            print(f"✅ 창의적 분석 리포트 저장: {output_path}")
        except Exception as e:
            print(f"❌ 리포트 저장 실패: {e}")
        
        # 요약 출력
        self._print_executive_summary(comprehensive_report['executive_summary'])
        
        return comprehensive_report
    
    def _generate_key_findings(self, results):
        """핵심 발견사항 생성"""
        if not results:
            return []
        
        # 변동계수 분포 분석
        vc_values = [r['creative_volatility_coefficient'] for r in results.values()]
        avg_vc = np.mean(vc_values)
        max_vc = max(vc_values)
        min_vc = min(vc_values)
        
        findings = [
            f"평균 창의적 변동계수: {avg_vc:.3f}",
            f"변동계수 범위: {min_vc:.3f} ~ {max_vc:.3f}",
            f"고위험 기업 비율: {sum(1 for vc in vc_values if vc >= 0.6) / len(vc_values) * 100:.1f}%"
        ]
        
        # DNA 타입 분포
        if hasattr(self, 'dna_profiles') and self.dna_profiles:
            dna_types = [profile['dna_type'] for profile in self.dna_profiles.values()]
            most_common_dna = max(set(dna_types), key=dna_types.count)
            findings.append(f"가장 흔한 DNA 타입: {most_common_dna}")
        
        # 디지털 성숙도 분포
        if hasattr(self, 'digital_transformation') and self.digital_transformation:
            digital_levels = [score['digital_maturity_level'] for score in self.digital_transformation.values()]
            advanced_digital = sum(1 for level in digital_levels if "Level 4" in level or "Level 5" in level)
            findings.append(f"디지털 선도 기업: {advanced_digital}개 ({advanced_digital/len(digital_levels)*100:.1f}%)")
        
        return findings
    
    def _analyze_risk_distribution(self, results):
        """위험 분포 분석"""
        if not results:
            return {}
        
        risk_levels = [r['risk_level'] for r in results.values()]
        risk_distribution = {}
        
        for risk in risk_levels:
            if risk in risk_distribution:
                risk_distribution[risk] += 1
            else:
                risk_distribution[risk] = 1
        
        # 비율로 변환
        total = len(risk_levels)
        for risk in risk_distribution:
            risk_distribution[risk] = {
                'count': risk_distribution[risk],
                'percentage': round(risk_distribution[risk] / total * 100, 1)
            }
        
        return risk_distribution
    
    def _summarize_digital_maturity(self):
        """디지털 성숙도 요약"""
        if not hasattr(self, 'digital_transformation') or not self.digital_transformation:
            return "디지털 성숙도 분석 데이터 없음"
        
        levels = list(self.digital_transformation.values())
        avg_score = np.mean([level['digital_transformation_score'] for level in levels])
        
        return f"평균 디지털 전환 점수: {avg_score:.3f}"
    
    def _summarize_business_health(self):
        """경영 건강도 요약"""
        if not hasattr(self, 'health_diagnostics') or not self.health_diagnostics:
            return "경영 건강도 분석 데이터 없음"
        
        health_scores = [diag['health_grade']['score'] for diag in self.health_diagnostics.values()]
        avg_health = np.mean(health_scores)
        
        return f"평균 경영 건강도: {avg_health:.3f}"
    
    def _generate_strategic_recommendations(self, results):
        """전략적 권장사항 생성"""
        if not results:
            return []
        
        high_risk_count = sum(1 for r in results.values() if r['creative_volatility_coefficient'] >= 0.6)
        total_count = len(results)
        
        recommendations = []
        
        if high_risk_count / total_count > 0.3:
            recommendations.extend([
                "전사적 리스크 관리 체계 구축 필요",
                "고위험 고객 집중 모니터링 시스템 도입",
                "예측 분석 기반 조기 경보 시스템 구축"
            ])
        
        recommendations.extend([
            "업종별 맞춤형 변동계수 기준 개발",
            "디지털 전환 지원 프로그램 확대",
            "에너지 효율성 컨설팅 서비스 강화",
            "데이터 기반 고객 세분화 전략 수립"
        ])
        
        return recommendations
    
    def _summarize_dna_insights(self):
        """DNA 분석 인사이트 요약"""
        if not hasattr(self, 'dna_profiles') or not self.dna_profiles:
            return {}
        
        return {
            'total_analyzed': len(self.dna_profiles),
            'unique_patterns_found': len(set(profile['dna_type'] for profile in self.dna_profiles.values())),
            'avg_uniqueness_score': round(np.mean([profile['uniqueness_score'] for profile in self.dna_profiles.values()]), 3)
        }
    
    def _summarize_health_results(self):
        """건강 진단 결과 요약"""
        if not hasattr(self, 'health_diagnostics') or not self.health_diagnostics:
            return {}
        
        health_grades = [diag['health_grade']['status'] for diag in self.health_diagnostics.values()]
        grade_distribution = {}
        for grade in health_grades:
            grade_distribution[grade] = grade_distribution.get(grade, 0) + 1
        
        return {
            'total_diagnosed': len(self.health_diagnostics),
            'grade_distribution': grade_distribution,
            'critical_patients': grade_distribution.get('critical', 0)
        }
    
    def _summarize_time_predictions(self):
        """시간여행 예측 요약"""
        if not hasattr(self, 'time_travel_predictions') or not self.time_travel_predictions:
            return {}
        
        return {
            'predictions_made': len(self.time_travel_predictions),
            'avg_continuity_score': round(np.mean([pred['time_continuity_score'] for pred in self.time_travel_predictions.values()]), 3)
        }
    
    def _summarize_expert_findings(self):
        """전문가 분석 요약"""
        if not hasattr(self, 'industry_experts') or not self.industry_experts:
            return {}
        
        total_recommendations = sum(len(expert['recommendations']) for expert in self.industry_experts.values())
        total_risks = sum(len(expert['risk_alerts']) for expert in self.industry_experts.values())
        
        return {
            'experts_deployed': len(self.industry_experts),
            'total_recommendations': total_recommendations,
            'total_risk_alerts': total_risks
        }
    
    def _summarize_digital_analysis(self):
        """디지털 분석 요약"""
        if not hasattr(self, 'digital_transformation') or not self.digital_transformation:
            return {}
        
        scores = list(self.digital_transformation.values())
        automation_avg = np.mean([s['automation_level'] for s in scores])
        smart_usage_avg = np.mean([s['smart_energy_usage'] for s in scores])
        
        return {
            'companies_analyzed': len(scores),
            'avg_automation_level': round(automation_avg, 3),
            'avg_smart_usage': round(smart_usage_avg, 3)
        }
    
    def _extract_immediate_actions(self, results):
        """즉시 실행 과제 추출"""
        immediate_actions = []
        
        for customer, result in results.items():
            if result['creative_volatility_coefficient'] >= 0.7:
                immediate_actions.append(f"{customer}: 긴급 모니터링 및 개입 필요")
        
        if not immediate_actions:
            immediate_actions = ["현재 긴급 대응이 필요한 고객 없음"]
        
        return immediate_actions
    
    def _extract_medium_term_strategy(self, results):
        """중기 전략 추출"""
        return [
            "업종별 변동계수 벤치마크 구축",
            "고객 세분화 기반 맞춤형 서비스 개발",
            "예측 분석 모델 고도화",
            "디지털 전환 지원 프로그램 확대"
        ]
    
    def _extract_long_term_vision(self, results):
        """장기 비전 추출"""
        return [
            "AI 기반 실시간 에너지 최적화 플랫폼 구축",
            "전력 사용 패턴 기반 신규 비즈니스 모델 개발",
            "산업별 에너지 효율성 벤치마킹 시스템 구축",
            "지속가능한 에너지 생태계 조성"
        ]
    
    def _print_executive_summary(self, summary):
        """경영진 요약 출력"""
        print("\n" + "="*80)
        print("🏆 창의적 전력 사용패턴 변동계수 분석 결과 요약")
        print("="*80)
        
        print("\n📊 핵심 발견사항:")
        for finding in summary['key_findings']:
            print(f"  • {finding}")
        
        print(f"\n🏥 경영 건강도: {summary['business_health_summary']}")
        print(f"🚀 디지털 성숙도: {summary['digital_maturity_overview']}")
        
        print("\n📋 전략적 권장사항:")
        for i, rec in enumerate(summary['strategic_recommendations'][:3], 1):
            print(f"  {i}. {rec}")
        
        print("\n🎯 분석 완료: 한국전력공사 창의적 변동계수 시스템 가동!")

# ============ 실행 함수 ============

def main_creative_analysis():
    """창의적 분석 메인 실행 함수"""
    print("🎨 한국전력공사 창의적 전력 사용패턴 변동계수 시스템")
    print("🧬 기업 경영활동 디지털 바이오마커 분석 시작")
    print("=" * 80)
    
    # 분석기 초기화
    analyzer = CreativePowerDNAAnalyzer()
    
    # 샘플 데이터 생성 (실제 환경에서는 LP 데이터 로딩)
    sample_lp_data = create_enhanced_sample_data()
    
    # 창의적 분석 실행
    report = analyzer.generate_creative_report(sample_lp_data)
    
    print("\n🎉 창의적 분석 완료!")
    print("📁 결과 파일: creative_volatility_report.json")
    
    return report

def create_enhanced_sample_data():
    """향상된 샘플 데이터 생성"""
    print("📊 샘플 데이터 생성 중...")
    
    data = []
    customers = [f'KEPCO_{i:04d}' for i in range(1, 11)]  # 10개 고객
    
    base_date = datetime(2024, 1, 1)
    
    for customer in customers:
        # 고객별 특성 부여
        customer_idx = int(customer.split('_')[1])
        
        # 기업 타입별 패턴 설정
        if customer_idx <= 3:
            # 제조업 패턴: 높은 사용량, 규칙적
            base_power = np.random.uniform(200, 500)
            volatility = np.random.uniform(0.1, 0.3)
        elif customer_idx <= 6:
            # 상업시설 패턴: 중간 사용량, 시간대별 변동
            base_power = np.random.uniform(100, 300)
            volatility = np.random.uniform(0.3, 0.6)
        else:
            # 서비스업 패턴: 낮은 사용량, 높은 변동성
            base_power = np.random.uniform(50, 200)
            volatility = np.random.uniform(0.5, 0.8)
        
        # 30일간 15분 간격 데이터 생성
        for day in range(30):
            for hour in range(24):
                for minute in [0, 15, 30, 45]:
                    timestamp = base_date + timedelta(days=day, hours=hour, minutes=minute)
                    
                    # 시간대별 패턴 반영
                    time_factor = 1.0
                    if 9 <= hour <= 18:  # 업무시간
                        time_factor = 1.2
                    elif 22 <= hour or hour <= 6:  # 야간
                        time_factor = 0.3
                    
                    # 요일별 패턴
                    weekday = timestamp.weekday()
                    if weekday >= 5:  # 주말
                        time_factor *= 0.6
                    
                    # 노이즈 추가
                    noise = np.random.normal(0, base_power * volatility)
                    power = max(0, base_power * time_factor + noise)
                    
                    data.append({
                        '대체고객번호': customer,
                        'LP수신일자': timestamp.strftime('%Y-%m-%d-%H:%M'),
                        '순방향유효전력': round(power, 2),
                        '지상무효': round(power * 0.1, 2),
                        '진상무효': round(power * 0.05, 2),
                        '피상전력': round(power * 1.1, 2)
                    })
    
    df = pd.DataFrame(data)
    print(f"✅ 샘플 데이터 생성 완료: {len(df):,}레코드")
    return df

if __name__ == "__main__":
    # 창의적 분석 실행
    main_creative_analysis()