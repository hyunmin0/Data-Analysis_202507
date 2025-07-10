import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from scipy import stats
from sklearn.preprocessing import StandardScaler
import warnings
import glob
import os
import json
import logging
import matplotlib

warnings.filterwarnings('ignore')
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
matplotlib.set_loglevel("ERROR")

# 한글 폰트 설정
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial Unicode MS', 'Malgun Gothic']
plt.rcParams['axes.unicode_minus'] = False

class KEPCOTimeSeriesAnalyzer:
    """한국전력공사 LP 데이터 시계열 패턴 분석 클래스"""
    
    def __init__(self, base_path='./'):
        """
        초기화
        Args:
            base_path: 데이터가 저장된 기본 경로
        """
        self.base_path = base_path
        self.customer_data = None
        self.lp_data = None
        self.analysis_results = {}
        
        # 결과 저장 디렉토리 생성
        self.output_dir = os.path.join(base_path, 'analysis_results')
        os.makedirs(self.output_dir, exist_ok=True)
        
        print("=" * 80)
        print("한국전력공사 전력 사용패턴 변동계수 개발 프로젝트")
        print("2단계: 시계열 패턴 분석 및 변동성 지표 개발")
        print("=" * 80)
        print(f"작업 디렉토리: {self.base_path}")
        print(f"결과 저장: {self.output_dir}")
        print()

    def load_customer_data(self, filename='제13회 산업부 공모전 대상고객.xlsx'):
        """실제 고객 기본정보 로딩"""
        print("🔄 1단계: 고객 기본정보 로딩...")
        
        try:
            file_path = os.path.join(self.base_path, filename)
            self.customer_data = pd.read_excel(file_path, header=1)
            
            print(f"✅ 고객 데이터 로딩 완료")
            print(f"   - 총 고객 수: {len(self.customer_data):,}명")
            print(f"   - 컬럼: {list(self.customer_data.columns)}")
            
            # 고객 분포 분석
            contract_dist = self.customer_data['계약종별'].value_counts()
            usage_dist = self.customer_data['사용용도'].value_counts()
            
            print(f"\n📊 고객 분포:")
            print(f"   - 계약종별: {len(contract_dist)}개 유형")
            print(f"   - 사용용도: {len(usage_dist)}개 유형")
            
            self.analysis_results['customer_summary'] = {
                'total_customers': len(self.customer_data),
                'contract_types': contract_dist.to_dict(),
                'usage_types': usage_dist.to_dict()
            }
            
            return True
            
        except Exception as e:
            print(f"❌ 고객 데이터 로딩 실패: {e}")
            return False

    def load_preprocessed_data(self):
        """실제 LP 데이터 로딩 (대용량 처리)"""
        print("\n🔄 2단계: LP 데이터 로딩...")
        
        try:
            analysis_results_path = './analysis_results/analysis_results.json'
            if os.path.exists(analysis_results_path):
                with open(analysis_results_path, 'r', encoding='utf-8') as f:
                    step1_rsults = json.load(f)
                print("1단계 결과 파일 확인")
            else:
                print("1단계 결과 파일 없음")
            
            processed_hdf5 = './analysis_results/processed_lp_data.h5'
            
            start_time = datetime.now()
            
            if os.path.exists(processed_hdf5):
                print("HDF5 파일 로딩")
                try:
                    self.lp_data = pd.read_hdf(processed_hdf5, key='df')
                    loading_method = "HDF5"
                    print(" 로딩 성공")
                except Exception as e:
                    print(f"로딩 실패{e}")
            
            else:
                print("전처리된 데이터 파일이 없습니다!")
                return False
            
            if 'datetime' in self.lp_data.columns:
                self.lp_data['datetime'] = pd.to_datetime(self.lp_data['datetime'])
            
            return True
        except Exception as e:
            import traceback
            traceback.print_exc()
            return False
        
    def load_external_data(self):
        print("\n 외부 데이터 로딩")
        
        try:
            weather_file = 'weather_daily_processed.csv'
            if os.path.exists(weather_file):
                self.weather_data = pd.read_csv(weather_file)
                self.weather_data['날짜'] = pd.to_datetime(self.weather_data['날짜'])
            else:
                print("기상 데이터 없음")
                self.weather_data = None
            
            calendar_file = 'power_analysis_calendar_2022_2025.csv'
            if os.path.exists(calendar_file):
                self.calendar_data = pd.read_csv(calendar_file)
                self.calendar_data['date'] = pd.to_datetime(self.calendar_data['date'])
                print("달력 데이터 있음")
            else:
                print("달력 데이터 없음")
                self.calendar_data = None
            return True
        except Exception as e:
            print("외부 데이터 로딩 실패")
            self.weather_data = None
            self.calendar_data = None
            return False


    def _validate_data_quality(self):
        """데이터 품질 검증"""
        print("\n🔍 데이터 품질 검증 중...")
        
        # 기본 통계
        numeric_columns = ['순방향 유효전력', '지상무효', '진상무효', '피상전력']
        available_numeric_cols = [col for col in numeric_columns if col in self.lp_data.columns]
        
        print(f"   📈 수치형 컬럼: {len(available_numeric_cols)}개")
        
        # 결측치 확인
        null_counts = self.lp_data[available_numeric_cols].isnull().sum()
        total_nulls = null_counts.sum()
        
        if total_nulls > 0:
            print(f"   ⚠️ 결측치: {total_nulls:,}개 ({total_nulls/len(self.lp_data)*100:.2f}%)")
            for col, count in null_counts.items():
                if count > 0:
                    print(f"      {col}: {count:,}개")
        else:
            print("   ✅ 결측치 없음")
        
        # 시간 간격 체크 (샘플 고객으로)
        sample_customers = self.lp_data['대체고객번호'].unique()[:3]
        
        print("   ⏰ 시간 간격 검증:")
        for customer in sample_customers:
            customer_data = self.lp_data[self.lp_data['대체고객번호'] == customer].sort_values('LP 수신일자')
            
            if len(customer_data) > 1:
                time_diffs = customer_data['LP 수신일자'].diff().dt.total_seconds() / 60
                time_diffs = time_diffs.dropna()
                
                if len(time_diffs) > 0:
                    avg_interval = time_diffs.mean()
                    std_interval = time_diffs.std()
                    print(f"      {customer}: 평균 {avg_interval:.1f}분 (표준편차: {std_interval:.1f})")
        
        # 분석 결과 저장
        self.analysis_results['lp_data_summary'] = {
            'total_records': len(self.lp_data),
            'customers': self.lp_data['대체고객번호'].nunique(),
            'null_counts': null_counts.to_dict(),
            'date_range': {
                'start': str(self.lp_data['LP 수신일자'].min()),
                'end': str(self.lp_data['LP 수신일자'].max())
            }
        }
        
        return True



    def analyze_temporal_patterns(self):
        """시계열 패턴 분석"""
        print("\n📈 3단계: 시계열 패턴 분석...")
        print("   🕐 시간 파생 변수 생성 중...")
        
        # datetime 컬럼 확인 및 변환
        if 'datetime' in self.lp_data.columns:
            datetime_col = 'datetime'
        elif 'LP 수신일자' in self.lp_data.columns:
            datetime_col = 'LP 수신일자'
        else:
            print("❌ 날짜/시간 컬럼을 찾을 수 없습니다")
            return False
        
        # datetime 타입 변환
        if not pd.api.types.is_datetime64_any_dtype(self.lp_data[datetime_col]):
            self.lp_data[datetime_col] = pd.to_datetime(self.lp_data[datetime_col], errors='coerce')
        
        # 파생 변수 생성
        try:
            self.lp_data['날짜'] = self.lp_data[datetime_col].dt.date
            self.lp_data['시간'] = self.lp_data[datetime_col].dt.hour
            self.lp_data['요일'] = self.lp_data[datetime_col].dt.weekday
            self.lp_data['월'] = self.lp_data[datetime_col].dt.month
            self.lp_data['주'] = self.lp_data[datetime_col].dt.isocalendar().week
            self.lp_data['주말여부'] = self.lp_data['요일'].isin([5, 6])
            
            print("   ✅ 시간 파생 변수 생성 완료")
        except Exception as e:
            print(f"❌ 시간 파생 변수 생성 실패: {e}")
            return False
        
        # 시간대별 패턴 분석
        print("   📊 시간대별 패턴 분석...")
        hourly_patterns = self.lp_data.groupby('시간')['순방향 유효전력'].agg([
            'mean', 'std', 'min', 'max', 'count'
        ]).round(2)
        
        avg_by_hour = hourly_patterns['mean']
        peak_threshold = avg_by_hour.quantile(0.75)
        off_peak_threshold = avg_by_hour.quantile(0.25)
        
        peak_hours = avg_by_hour[avg_by_hour >= peak_threshold].index.tolist()
        off_peak_hours = avg_by_hour[avg_by_hour <= off_peak_threshold].index.tolist()
        
        # 요일별 패턴 분석
        print("   📅 요일별 패턴 분석...")
        daily_patterns = self.lp_data.groupby('요일')['순방향 유효전력'].agg([
            'mean', 'std', 'count'
        ]).round(2)
        
        weekday_avg = self.lp_data[~self.lp_data['주말여부']]['순방향 유효전력'].mean()
        weekend_avg = self.lp_data[self.lp_data['주말여부']]['순방향 유효전력'].mean()
        weekend_ratio = weekend_avg / weekday_avg if weekday_avg > 0 else 0
        
        # 월별 계절성 패턴
        print("   🗓️ 월별 계절성 분석...")
        monthly_patterns = self.lp_data.groupby('월')['순방향 유효전력'].agg([
            'mean', 'std', 'count'
        ]).round(2)
        
        season_map = {12: '겨울', 1: '겨울', 2: '겨울',
                     3: '봄', 4: '봄', 5: '봄',
                     6: '여름', 7: '여름', 8: '여름',
                     9: '가을', 10: '가을', 11: '가을'}
        
        self.lp_data['계절'] = self.lp_data['월'].map(season_map)
        seasonal_patterns = self.lp_data.groupby('계절')['순방향 유효전력'].agg([
            'mean', 'std', 'count'
        ]).round(2)
        
        # 분석 결과 저장
        self.analysis_results['temporal_patterns'] = {
            'hourly_patterns': hourly_patterns.to_dict(),
            'daily_patterns': daily_patterns.to_dict(),
            'monthly_patterns': monthly_patterns.to_dict(),
            'seasonal_patterns': seasonal_patterns.to_dict(),
            'peak_hours': peak_hours,
            'off_peak_hours': off_peak_hours,
            'weekend_ratio': weekend_ratio
        }
        
        return True

    def analyze_volatility_indicators(self):
        """변동성 지표 분석 (집계 중심)"""
        print("\n📊 4단계: 변동성 지표 분석...")
        
        customers = self.lp_data['대체고객번호'].unique()
        print(f"   🔄 {len(customers)}명 고객 변동성 분석 중...")
        
        # 전체 데이터에 대한 집계 분석
        
        # 1. 전체 변동성 통계
        overall_power = self.lp_data['순방향 유효전력']
        overall_cv = overall_power.std() / overall_power.mean() if overall_power.mean() > 0 else 0
        
        print(f"   📈 전체 데이터 변동성:")
        print(f"      전체 변동계수: {overall_cv:.4f}")
        print(f"      평균 전력: {overall_power.mean():.2f}kW")
        print(f"      표준편차: {overall_power.std():.2f}kW")
        
        # 2. 시간대별 변동성 패턴
        hourly_volatility = self.lp_data.groupby('시간')['순방향 유효전력'].agg([
            'mean', 'std', 'count'
        ])
        hourly_volatility['cv'] = hourly_volatility['std'] / hourly_volatility['mean']
        
        print(f"\n   ⏰ 시간대별 변동성 패턴:")
        high_volatility_hours = hourly_volatility.nlargest(3, 'cv').index.tolist()
        low_volatility_hours = hourly_volatility.nsmallest(3, 'cv').index.tolist()
        print(f"      고변동성 시간대: {high_volatility_hours}시 (CV: {hourly_volatility.loc[high_volatility_hours, 'cv'].mean():.4f})")
        print(f"      저변동성 시간대: {low_volatility_hours}시 (CV: {hourly_volatility.loc[low_volatility_hours, 'cv'].mean():.4f})")
        
        # 3. 요일별 변동성 패턴
        daily_volatility = self.lp_data.groupby('요일')['순방향 유효전력'].agg([
            'mean', 'std', 'count'
        ])
        daily_volatility['cv'] = daily_volatility['std'] / daily_volatility['mean']
        
        weekday_cv = daily_volatility.loc[0:4, 'cv'].mean()  # 월-금
        weekend_cv = daily_volatility.loc[5:6, 'cv'].mean()  # 토-일
        
        print(f"\n   📅 요일별 변동성 패턴:")
        print(f"      평일 평균 변동계수: {weekday_cv:.4f}")
        print(f"      주말 평균 변동계수: {weekend_cv:.4f}")
        print(f"      주말/평일 변동성 비율: {weekend_cv/weekday_cv:.3f}")
        
        # 4. 월별 변동성 패턴
        monthly_volatility = self.lp_data.groupby('월')['순방향 유효전력'].agg([
            'mean', 'std', 'count'
        ])
        monthly_volatility['cv'] = monthly_volatility['std'] / monthly_volatility['mean']
        
        print(f"\n   🗓️ 월별 변동성 패턴:")
        high_var_months = monthly_volatility.nlargest(2, 'cv').index.tolist()
        low_var_months = monthly_volatility.nsmallest(2, 'cv').index.tolist()
        print(f"      고변동성 월: {high_var_months}월")
        print(f"      저변동성 월: {low_var_months}월")
        
        # 5. 고객별 변동성 분포 (요약 통계만)
        print(f"\n   👥 고객별 변동성 분포 분석...")
        
        # 청크 단위로 고객별 변동계수 계산 (메모리 효율성)
        chunk_size = 100
        customer_cvs = []
        
        for i in range(0, len(customers), chunk_size):
            chunk_customers = customers[i:i+chunk_size]
            if (i // chunk_size + 1) % 5 == 0:  # 500명마다 진행상황 출력
                print(f"      진행: {min(i+chunk_size, len(customers))}/{len(customers)} ({min(i+chunk_size, len(customers))/len(customers)*100:.1f}%)")
            
            for customer in chunk_customers:
                customer_data = self.lp_data[self.lp_data['대체고객번호'] == customer]
                power_series = customer_data['순방향 유효전력']
                
                if len(power_series) >= 96 and power_series.mean() > 0:  # 최소 1일 데이터
                    cv = power_series.std() / power_series.mean()
                    customer_cvs.append(cv)
        
        # 고객별 변동계수 분포 통계
        cv_array = np.array(customer_cvs)
        cv_percentiles = np.percentile(cv_array, [10, 25, 50, 75, 90])
        
        print(f"   📊 고객별 변동계수 분포 ({len(customer_cvs)}명):")
        print(f"      평균: {cv_array.mean():.4f}")
        print(f"      표준편차: {cv_array.std():.4f}")
        print(f"      10%ile: {cv_percentiles[0]:.4f}")
        print(f"      25%ile: {cv_percentiles[1]:.4f}")
        print(f"      50%ile: {cv_percentiles[2]:.4f}")
        print(f"      75%ile: {cv_percentiles[3]:.4f}")
        print(f"      90%ile: {cv_percentiles[4]:.4f}")
        
        # 변동계수 구간별 고객 수
        cv_bins = [0, 0.1, 0.2, 0.3, 0.5, 1.0, float('inf')]
        cv_labels = ['매우 안정 (<0.1)', '안정 (0.1-0.2)', '보통 (0.2-0.3)', 
                    '높음 (0.3-0.5)', '매우 높음 (0.5-1.0)', '극히 높음 (>1.0)']
        
        cv_counts = pd.cut(cv_array, bins=cv_bins, labels=cv_labels, include_lowest=True).value_counts()
        
        print(f"\n   🎯 변동성 등급별 고객 분포:")
        for grade, count in cv_counts.items():
            percentage = count / len(customer_cvs) * 100
            print(f"      {grade}: {count}명 ({percentage:.1f}%)")
        
        # 분석 결과 저장
        self.analysis_results['volatility_analysis'] = {
            'overall_cv': overall_cv,
            'hourly_volatility': hourly_volatility.to_dict(),
            'daily_volatility': daily_volatility.to_dict(),
            'monthly_volatility': monthly_volatility.to_dict(),
            'customer_cv_stats': {
                'count': len(customer_cvs),
                'mean': float(cv_array.mean()),
                'std': float(cv_array.std()),
                'percentiles': {
                    '10%': float(cv_percentiles[0]),
                    '25%': float(cv_percentiles[1]),
                    '50%': float(cv_percentiles[2]),
                    '75%': float(cv_percentiles[3]),
                    '90%': float(cv_percentiles[4])
                }
            },
            'volatility_distribution': cv_counts.to_dict()
        }
        
        # 요약 데이터만 CSV로 저장 (개별 고객 데이터는 제외)
        summary_data = {
            'metric': ['overall_cv', 'weekday_cv', 'weekend_cv', 'customer_cv_mean', 
                      'customer_cv_std', 'customer_cv_median'],
            'value': [overall_cv, weekday_cv, weekend_cv, cv_array.mean(), 
                     cv_array.std(), cv_percentiles[2]]
        }
        
        summary_df = pd.DataFrame(summary_data)
        output_file = os.path.join(self.output_dir, 'volatility_summary.csv')
        summary_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\n   💾 변동성 요약 저장: {output_file}")
        
        return cv_array

    def detect_anomalies(self):
        """이상 패턴 탐지 (집계 중심)"""
        print("\n🚨 5단계: 이상 패턴 탐지...")
        
        customers = self.lp_data['대체고객번호'].unique()
        print(f"   🔍 {len(customers)}명 고객 이상 패턴 탐지 중...")
        
        # 전체 데이터 기반 이상 패턴 탐지
        
        # 1. 전체 데이터의 통계적 이상치 임계값 설정
        overall_power = self.lp_data['순방향 유효전력']
        q1, q3 = overall_power.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        # 전체 통계적 이상치
        total_outliers = ((overall_power < lower_bound) | (overall_power > upper_bound)).sum()
        outlier_rate = total_outliers / len(overall_power) * 100
        
        print(f"   📊 전체 데이터 이상치 현황:")
        print(f"      통계적 이상치: {total_outliers:,}개 ({outlier_rate:.2f}%)")
        print(f"      정상 범위: {lower_bound:.1f} ~ {upper_bound:.1f}kW")
        
        # 2. 시간대별 이상 패턴
        night_hours = [0, 1, 2, 3, 4, 5]  # 야간 시간대
        day_hours = [9, 10, 11, 12, 13, 14, 15, 16, 17]  # 주간 시간대
        
        night_data = self.lp_data[self.lp_data['시간'].isin(night_hours)]
        day_data = self.lp_data[self.lp_data['시간'].isin(day_hours)]
        
        night_avg = night_data['순방향 유효전력'].mean()
        day_avg = day_data['순방향 유효전력'].mean()
        night_day_ratio = night_avg / day_avg if day_avg > 0 else 0
        
        print(f"\n   🌙 시간대별 사용 패턴:")
        print(f"      야간 평균: {night_avg:.2f}kW")
        print(f"      주간 평균: {day_avg:.2f}kW")
        print(f"      야간/주간 비율: {night_day_ratio:.3f}")
        
        # 3. 0값 패턴 분석
        zero_count = (overall_power == 0).sum()
        zero_rate = zero_count / len(overall_power) * 100
        
        print(f"\n   ⚫ 0값 패턴 분석:")
        print(f"      0값 측정: {zero_count:,}개 ({zero_rate:.2f}%)")
        
        # 4. 급격한 변화 패턴 (전체 데이터 기준)
        power_changes = self.lp_data.sort_values(['대체고객번호', 'LP 수신일자'])['순방향 유효전력'].pct_change().abs()
        sudden_changes = power_changes[power_changes > 2.0]  # 200% 이상 변화
        sudden_change_rate = len(sudden_changes) / len(power_changes.dropna()) * 100
        
        print(f"\n   ⚡ 급격한 변화 패턴:")
        print(f"      급격한 변화: {len(sudden_changes):,}건 ({sudden_change_rate:.2f}%)")
        
        # 5. 고객별 이상 패턴 요약 통계 (개별 출력 없이)
        anomaly_customers = {
            'high_night_usage': 0,      # 야간 과다 사용
            'excessive_zeros': 0,        # 과도한 0값
            'high_volatility': 0,        # 높은 변동성
            'statistical_outliers': 0    # 통계적 이상치 다수
        }
        
        chunk_size = 100
        processed_customers = 0
        
        for i in range(0, len(customers), chunk_size):
            chunk_customers = customers[i:i+chunk_size]
            if (i // chunk_size + 1) % 5 == 0:
                print(f"      진행: {min(i+chunk_size, len(customers))}/{len(customers)} ({min(i+chunk_size, len(customers))/len(customers)*100:.1f}%)")
            
            for customer in chunk_customers:
                customer_data = self.lp_data[self.lp_data['대체고객번호'] == customer]
                power_series = customer_data['순방향 유효전력']
                
                if len(power_series) < 96:  # 최소 1일 데이터 필요
                    continue
                
                processed_customers += 1
                
                # 야간 과다 사용 체크
                customer_night = customer_data[customer_data['시간'].isin(night_hours)]['순방향 유효전력'].mean()
                customer_day = customer_data[customer_data['시간'].isin(day_hours)]['순방향 유효전력'].mean()
                if customer_day > 0 and customer_night / customer_day > 0.8:
                    anomaly_customers['high_night_usage'] += 1
                
                # 과도한 0값 체크
                zero_ratio = (power_series == 0).sum() / len(power_series)
                if zero_ratio > 0.1:  # 10% 이상이 0값
                    anomaly_customers['excessive_zeros'] += 1
                
                # 높은 변동성 체크
                if power_series.mean() > 0:
                    cv = power_series.std() / power_series.mean()
                    if cv > 1.0:  # 변동계수 1.0 이상
                        anomaly_customers['high_volatility'] += 1
                
                # 통계적 이상치 다수 체크
                customer_outliers = ((power_series < lower_bound) | (power_series > upper_bound)).sum()
                outlier_ratio = customer_outliers / len(power_series)
                if outlier_ratio > 0.05:  # 5% 이상이 이상치
                    anomaly_customers['statistical_outliers'] += 1
        
        # 종합 이상 패턴 고객 (중복 제거를 위해 실제로는 근사치)
        total_anomaly_customers = max(anomaly_customers.values())  # 단순 근사
        anomaly_rate = total_anomaly_customers / processed_customers * 100 if processed_customers > 0 else 0
        
        print(f"\n   📊 이상 패턴 고객 요약 ({processed_customers}명 분석):")
        print(f"      야간 과다 사용: {anomaly_customers['high_night_usage']}명")
        print(f"      과도한 0값: {anomaly_customers['excessive_zeros']}명")
        print(f"      높은 변동성: {anomaly_customers['high_volatility']}명")
        print(f"      통계적 이상치 다수: {anomaly_customers['statistical_outliers']}명")
        print(f"      전체 이상 패턴 비율: 약 {anomaly_rate:.1f}%")
        
        # 분석 결과 저장
        self.analysis_results['anomaly_analysis'] = {
            'processed_customers': processed_customers,
            'total_outliers': int(total_outliers),
            'outlier_rate': float(outlier_rate),
            'zero_count': int(zero_count),
            'zero_rate': float(zero_rate),
            'sudden_changes': len(sudden_changes),
            'sudden_change_rate': float(sudden_change_rate),
            'night_day_ratio': float(night_day_ratio),
            'anomaly_customers': anomaly_customers,
            'estimated_anomaly_rate': float(anomaly_rate)
        }
        
        return anomaly_customers


    def create_summary_visualizations(self):
        """요약 시각화 생성 (집계 데이터 중심)"""
        print("\n📊 6단계: 분석 결과 시각화...")
        
        try:
            # 1. 시간대별/요일별 패턴 시각화
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # 시간대별 패턴
            hourly_avg = self.lp_data.groupby('시간')['순방향 유효전력'].mean()
            axes[0, 0].plot(hourly_avg.index, hourly_avg.values, marker='o', linewidth=2, color='blue')
            axes[0, 0].set_title('시간대별 평균 전력 사용량', fontsize=14, fontweight='bold')
            axes[0, 0].set_xlabel('시간')
            axes[0, 0].set_ylabel('평균 유효전력 (kW)')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].set_xticks(range(0, 24, 3))
            
            # 요일별 패턴
            daily_avg = self.lp_data.groupby('요일')['순방향 유효전력'].mean()
            weekday_names = ['월', '화', '수', '목', '금', '토', '일']
            axes[0, 1].bar(range(len(daily_avg)), daily_avg.values, color='skyblue')
            axes[0, 1].set_title('요일별 평균 전력 사용량', fontsize=14, fontweight='bold')
            axes[0, 1].set_xlabel('요일')
            axes[0, 1].set_ylabel('평균 유효전력 (kW)')
            axes[0, 1].set_xticks(range(7))
            axes[0, 1].set_xticklabels(weekday_names)
            axes[0, 1].grid(True, alpha=0.3, axis='y')
            
            # 시간대별 변동성
            hourly_std = self.lp_data.groupby('시간')['순방향 유효전력'].std()
            axes[1, 0].plot(hourly_std.index, hourly_std.values, marker='s', linewidth=2, color='red')
            axes[1, 0].set_title('시간대별 전력 사용량 변동성', fontsize=14, fontweight='bold')
            axes[1, 0].set_xlabel('시간')
            axes[1, 0].set_ylabel('표준편차 (kW)')
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].set_xticks(range(0, 24, 3))
            
            # 월별 계절성 패턴
            monthly_avg = self.lp_data.groupby('월')['순방향 유효전력'].mean()
            axes[1, 1].plot(monthly_avg.index, monthly_avg.values, marker='s', linewidth=2, color='orange')
            axes[1, 1].set_title('월별 평균 전력 사용량 (계절성)', fontsize=14, fontweight='bold')
            axes[1, 1].set_xlabel('월')
            axes[1, 1].set_ylabel('평균 유효전력 (kW)')
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].set_xticks(range(1, 13))
            
            plt.tight_layout()
            
            # 이미지 저장
            output_file = os.path.join(self.output_dir, 'temporal_patterns_summary.png')
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"   💾 시계열 패턴 시각화 저장: {output_file}")
            
            # 2. 변동성 및 이상치 분포 시각화
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # 전체 전력 사용량 분포
            axes[0, 0].hist(self.lp_data['순방향 유효전력'], bins=50, alpha=0.7, color='lightblue')
            axes[0, 0].set_title('전력 사용량 분포', fontsize=14, fontweight='bold')
            axes[0, 0].set_xlabel('순방향 유효전력 (kW)')
            axes[0, 0].set_ylabel('빈도')
            axes[0, 0].grid(True, alpha=0.3, axis='y')
            
            # 시간대별 변동계수
            hourly_volatility = self.analysis_results.get('volatility_analysis', {}).get('hourly_volatility', {})
            if hourly_volatility and 'cv' in hourly_volatility:
                cv_data = hourly_volatility['cv']
                hours = list(cv_data.keys())
                cv_values = list(cv_data.values())
                axes[0, 1].bar(hours, cv_values, color='lightgreen')
                axes[0, 1].set_title('시간대별 변동계수', fontsize=14, fontweight='bold')
                axes[0, 1].set_xlabel('시간')
                axes[0, 1].set_ylabel('변동계수')
                axes[0, 1].grid(True, alpha=0.3, axis='y')
            
            # 요일별 변동계수
            daily_volatility = self.analysis_results.get('volatility_analysis', {}).get('daily_volatility', {})
            if daily_volatility and 'cv' in daily_volatility:
                cv_data = daily_volatility['cv']
                weekdays = list(cv_data.keys())
                cv_values = list(cv_data.values())
                weekday_names = ['월', '화', '수', '목', '금', '토', '일']
                axes[1, 0].bar(range(len(cv_values)), cv_values, color='purple')
                axes[1, 0].set_title('요일별 변동계수', fontsize=14, fontweight='bold')
                axes[1, 0].set_xlabel('요일')
                axes[1, 0].set_ylabel('변동계수')
                axes[1, 0].set_xticks(range(7))
                axes[1, 0].set_xticklabels(weekday_names)
                axes[1, 0].grid(True, alpha=0.3, axis='y')
            
            # 월별 변동계수
            monthly_volatility = self.analysis_results.get('volatility_analysis', {}).get('monthly_volatility', {})
            if monthly_volatility and 'cv' in monthly_volatility:
                cv_data = monthly_volatility['cv']
                months = list(cv_data.keys())
                cv_values = list(cv_data.values())
                axes[1, 1].plot(months, cv_values, marker='o', linewidth=2, color='red')
                axes[1, 1].set_title('월별 변동계수 (계절성)', fontsize=14, fontweight='bold')
                axes[1, 1].set_xlabel('월')
                axes[1, 1].set_ylabel('변동계수')
                axes[1, 1].grid(True, alpha=0.3)
                axes[1, 1].set_xticks(range(1, 13))
            
            plt.tight_layout()
            
            # 이미지 저장
            output_file = os.path.join(self.output_dir, 'volatility_analysis_summary.png')
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"   💾 변동성 분석 시각화 저장: {output_file}")
            
            # 3. 변동성 등급별 분포 시각화 (있는 경우)
            volatility_dist = self.analysis_results.get('volatility_analysis', {}).get('volatility_distribution', {})
            if volatility_dist:
                fig, ax = plt.subplots(1, 1, figsize=(12, 8))
                
                grades = list(volatility_dist.keys())
                counts = list(volatility_dist.values())
                
                bars = ax.bar(range(len(grades)), counts, color=['green', 'lightgreen', 'yellow', 'orange', 'red', 'darkred'])
                ax.set_title('변동성 등급별 고객 분포', fontsize=16, fontweight='bold')
                ax.set_xlabel('변동성 등급', fontsize=12)
                ax.set_ylabel('고객 수', fontsize=12)
                ax.set_xticks(range(len(grades)))
                ax.set_xticklabels(grades, rotation=45, ha='right')
                ax.grid(True, alpha=0.3, axis='y')
                
                # 각 막대 위에 수치 표시
                for i, (bar, count) in enumerate(zip(bars, counts)):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                           f'{count}명', ha='center', va='bottom', fontweight='bold')
                
                plt.tight_layout()
                
                # 이미지 저장
                output_file = os.path.join(self.output_dir, 'volatility_distribution.png')
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"   💾 변동성 분포 시각화 저장: {output_file}")
            
            return True
            
        except Exception as e:
            print(f"   ❌ 시각화 생성 실패: {e}")
            import traceback
            traceback.print_exc()
            return False

    def create_summary_visualizations(self):
        """요약 시각화 생성"""
        print("\n📊 6단계: 분석 결과 시각화...")
        
        try:
            # 1. 시간대별 평균 전력 사용 패턴
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # 시간대별 패턴
            hourly_avg = self.lp_data.groupby('시간')['순방향 유효전력'].mean()
            axes[0, 0].plot(hourly_avg.index, hourly_avg.values, marker='o', linewidth=2)
            axes[0, 0].set_title('시간대별 평균 전력 사용량', fontsize=14, fontweight='bold')
            axes[0, 0].set_xlabel('시간')
            axes[0, 0].set_ylabel('평균 유효전력 (kW)')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].set_xticks(range(0, 24, 3))
            
            # 요일별 패턴
            daily_avg = self.lp_data.groupby('요일')['순방향 유효전력'].mean()
            weekday_names = ['월', '화', '수', '목', '금', '토', '일']
            axes[0, 1].bar(range(len(daily_avg)), daily_avg.values, color='skyblue')
            axes[0, 1].set_title('요일별 평균 전력 사용량', fontsize=14, fontweight='bold')
            axes[0, 1].set_xlabel('요일')
            axes[0, 1].set_ylabel('평균 유효전력 (kW)')
            axes[0, 1].set_xticks(range(7))
            axes[0, 1].set_xticklabels(weekday_names)
            axes[0, 1].grid(True, alpha=0.3, axis='y')
            
            # 변동계수 분포 (변동성 분석이 완료된 경우)
            if 'volatility_analysis' in self.analysis_results:
                volatility_file = os.path.join(self.output_dir, 'volatility_indicators.csv')
                if os.path.exists(volatility_file):
                    volatility_df = pd.read_csv(volatility_file)
                    axes[1, 0].hist(volatility_df['cv_basic'].dropna(), bins=30, alpha=0.7, color='lightgreen')
                    axes[1, 0].set_title('변동계수 분포', fontsize=14, fontweight='bold')
                    axes[1, 0].set_xlabel('변동계수 (CV)')
                    axes[1, 0].set_ylabel('고객 수')
                    axes[1, 0].grid(True, alpha=0.3, axis='y')
            
            # 월별 계절성 패턴
            monthly_avg = self.lp_data.groupby('월')['순방향 유효전력'].mean()
            month_names = ['1월', '2월', '3월', '4월', '5월', '6월', 
                          '7월', '8월', '9월', '10월', '11월', '12월']
            axes[1, 1].plot(monthly_avg.index, monthly_avg.values, marker='s', linewidth=2, color='orange')
            axes[1, 1].set_title('월별 평균 전력 사용량 (계절성)', fontsize=14, fontweight='bold')
            axes[1, 1].set_xlabel('월')
            axes[1, 1].set_ylabel('평균 유효전력 (kW)')
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].set_xticks(range(1, 13))
            
            plt.tight_layout()
            
            # 이미지 저장
            output_file = os.path.join(self.output_dir, 'temporal_patterns_summary.png')
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"   💾 시계열 패턴 시각화 저장: {output_file}")
            
            # 2. 변동성 관련 시각화 (추가)
            if 'volatility_analysis' in self.analysis_results:
                volatility_file = os.path.join(self.output_dir, 'volatility_indicators.csv')
                if os.path.exists(volatility_file):
                    volatility_df = pd.read_csv(volatility_file)
                    
                    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
                    
                    # 평균 사용량 vs 변동계수
                    axes[0, 0].scatter(volatility_df['mean_power'], volatility_df['cv_basic'], alpha=0.6, s=20)
                    axes[0, 0].set_title('평균 사용량 vs 변동계수', fontsize=14, fontweight='bold')
                    axes[0, 0].set_xlabel('평균 전력 (kW)')
                    axes[0, 0].set_ylabel('변동계수')
                    axes[0, 0].grid(True, alpha=0.3)
                    
                    # 시간대별 변동성 vs 일별 변동성
                    axes[0, 1].scatter(volatility_df['hourly_cv_mean'], volatility_df['daily_cv_mean'], alpha=0.6, s=20, color='red')
                    axes[0, 1].set_title('시간대별 vs 일별 변동성', fontsize=14, fontweight='bold')
                    axes[0, 1].set_xlabel('시간대별 평균 변동계수')
                    axes[0, 1].set_ylabel('일별 평균 변동계수')
                    axes[0, 1].grid(True, alpha=0.3)
                    
                    # 주말/평일 변동계수 비교
                    weekend_weekday_ratio = volatility_df['weekend_weekday_cv_ratio'].dropna()
                    axes[1, 0].hist(weekend_weekday_ratio, bins=20, alpha=0.7, color='purple')
                    axes[1, 0].set_title('주말/평일 변동계수 비율 분포', fontsize=14, fontweight='bold')
                    axes[1, 0].set_xlabel('주말/평일 변동계수 비율')
                    axes[1, 0].set_ylabel('고객 수')
                    axes[1, 0].grid(True, alpha=0.3, axis='y')
                    
                    # 변동계수 상위/하위 분포
                    cv_top10 = volatility_df.nlargest(10, 'cv_basic')['cv_basic']
                    cv_bottom10 = volatility_df.nsmallest(10, 'cv_basic')['cv_basic']
                    
                    x_pos = range(10)
                    width = 0.35
                    axes[1, 1].bar([x - width/2 for x in x_pos], cv_top10.values, width, 
                                  label='상위 10명', alpha=0.8, color='red')
                    axes[1, 1].bar([x + width/2 for x in x_pos], cv_bottom10.values, width, 
                                  label='하위 10명', alpha=0.8, color='blue')
                    axes[1, 1].set_title('변동계수 상위/하위 10명 비교', fontsize=14, fontweight='bold')
                    axes[1, 1].set_xlabel('순위')
                    axes[1, 1].set_ylabel('변동계수')
                    axes[1, 1].legend()
                    axes[1, 1].grid(True, alpha=0.3, axis='y')
                    
                    plt.tight_layout()
                    
                    # 이미지 저장
                    output_file = os.path.join(self.output_dir, 'volatility_analysis_summary.png')
                    plt.savefig(output_file, dpi=300, bbox_inches='tight')
                    plt.close()
                    print(f"   💾 변동성 분석 시각화 저장: {output_file}")
            
            return True
            
        except Exception as e:
            print(f"   ❌ 시각화 생성 실패: {e}")
            return False
        

    def save_analysis_results(self):
        """분석 결과를 JSON 파일로 저장"""
        print("\n💾 8단계: 분석 결과 저장...")
        
        try:
            # JSON으로 저장 가능한 형태로 변환
            results_for_json = {}
            
            for key, value in self.analysis_results.items():
                if isinstance(value, dict):
                    results_for_json[key] = {}
                    for sub_key, sub_value in value.items():
                        if hasattr(sub_value, 'to_dict'):  # pandas 객체인 경우
                            results_for_json[key][sub_key] = sub_value.to_dict()
                        else:
                            results_for_json[key][sub_key] = sub_value
                else:
                    results_for_json[key] = value
            
            # JSON 파일로 저장
            output_file = os.path.join(self.output_dir, 'analysis_results2.json')
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results_for_json, f, ensure_ascii=False, indent=2, default=str)
            
            print(f"   💾 분석 결과 JSON 저장: {output_file}")
            return True
            
        except Exception as e:
            print(f"   ❌ 분석 결과 저장 실패: {e}")
            return False

    def run_complete_analysis(self):
        """전체 분석 프로세스 실행"""
        start_time = datetime.now()
        
        print("🚀 한국전력공사 LP 데이터 시계열 패턴 분석 시작")
        print(f"시작 시간: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        try:
            # 1. 전처리된 데이터 로딩
            if not self.load_preprocessed_data():
                print("❌ 전처리된 데이터 로딩 실패로 분석을 중단합니다.")
                return False
            
            # 2. 외부 데이터 로딩
            self.load_external_data()
            
            # 3. 시계열 패턴 분석
            if not self.analyze_temporal_patterns():
                print("❌ 시계열 패턴 분석 실패")
                return False
            
            # 4. 변동성 지표 분석
            cv_array = self.analyze_volatility_indicators()
            if cv_array is None or len(cv_array) == 0:
                print("❌ 변동성 지표 분석 실패")
                return False
            
            # 5. 이상 패턴 탐지
            anomaly_summary = self.detect_anomalies()
            if anomaly_summary is None:
                print("❌ 이상 패턴 탐지 실패")
                return False
            
            # 6. 시각화 생성
            self.create_summary_visualizations()
            
            
            # 7. 결과 저장
            self.save_analysis_results()
            
            end_time = datetime.now()
            duration = end_time - start_time
            
            print("\n" + "=" * 80)
            print("🎉 시계열 패턴 분석 완료!")
            print("=" * 80)
            print(f"소요 시간: {duration}")
            print(f"결과 저장 위치: {self.output_dir}")
            
            return True
            
        except Exception as e:
            print(f"\n❌ 분석 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            return False

if __name__ == "__main__":
    analyzer = KEPCOTimeSeriesAnalyzer()
    
    success = analyzer.run_complete_analysis()
    
    if success:
        print("성공")