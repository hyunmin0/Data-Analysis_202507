import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 폰트 설정 (에러 방지)
plt.rcParams['axes.unicode_minus'] = False

class KepcoDataPreprocessor:
    """
    한국전력공사 LP 데이터 전처리 및 탐색적 분석
    
    단계별 분석 계획:
    1단계: 데이터 품질 점검 (30분)
    2단계: 기본 패턴 탐색 (60분) 
    3단계: 변동성 기초 분석 (90분)
    4단계: 이상 패턴 탐지 (60분)
    5단계: 전처리 방향 결정 (30분)
    """
    
    def __init__(self):
        self.data_quality_report = {}
        self.pattern_analysis = {}
        self.variability_analysis = {}
    
    # ============ 데이터 로딩 및 결합 ============
    
    def load_and_combine_lp_data(self, lp_files):
        """
        LP 데이터 파일들을 읽어서 결합
        파일들은 한 달을 반으로 나눠서 제공됨 (예: LP데이터1.csv + LP데이터2.csv)
        """
        print("📂 LP 데이터 파일들 로딩 및 결합 중...")
        
        combined_data = []
        
        for i, file_path in enumerate(lp_files):
            try:
                df = pd.read_csv(file_path)
                print(f"  ✅ {file_path}: {len(df):,}건 로딩")
                
                # 기본 정보 출력
                if 'LP수신일자' in df.columns:
                    dates = pd.to_datetime(df['LP수신일자'])
                    print(f"     기간: {dates.min()} ~ {dates.max()}")
                    print(f"     고객수: {df['대체고객번호'].nunique()}명")
                
                combined_data.append(df)
                
            except Exception as e:
                print(f"  ❌ {file_path} 로딩 실패: {e}")
        
        if not combined_data:
            raise ValueError("로딩된 LP 데이터가 없습니다.")
        
        # 데이터 결합
        final_data = pd.concat(combined_data, ignore_index=True)
        
        print(f"  🔗 결합 완료: 총 {len(final_data):,}건")
        print(f"     전체 기간: {pd.to_datetime(final_data['LP수신일자']).min()} ~ {pd.to_datetime(final_data['LP수신일자']).max()}")
        print(f"     총 고객수: {final_data['대체고객번호'].nunique()}명")
        
        return final_data
    
    # ============ 1단계: 데이터 품질 점검 (30분) ============
    
    def check_data_quality(self, lp_data, customer_data):
        """
        데이터 품질 점검 및 기본 정보 분석
        """
        print("🔍 1단계: 데이터 품질 점검 시작...")
        
        # 고객 기본정보 분석
        customer_info = self._analyze_customer_info(customer_data)
        
        # LP 데이터 품질 점검
        lp_quality = self._check_lp_data_quality(lp_data)
        
        self.data_quality_report = {
            'customer_info': customer_info,
            'lp_quality': lp_quality,
            'data_completeness': self._calculate_completeness(lp_data),
            'anomaly_detection': self._detect_data_anomalies(lp_data)
        }
        
        self._print_quality_summary()
        return self.data_quality_report
    
    def _analyze_customer_info(self, customer_data):
        """고객 기본정보 분석"""
        if customer_data is None:
            return {"message": "고객 데이터 없음"}
        
        print(f"  📋 고객 데이터 컬럼: {list(customer_data.columns)}")
        
        # 가능한 컬럼명들 매핑
        column_mapping = {
            '계약종별': ['계약종별', 'contract_type', 'Contract_Type'],
            '사용용도': ['사용용도', 'usage_purpose', 'Usage_Purpose'], 
            '산업분류': ['산업분류', 'industry', 'Industry']
        }
        
        info = {'total_customers': len(customer_data)}
        
        for key, possible_cols in column_mapping.items():
            found_col = None
            for col in possible_cols:
                if col in customer_data.columns:
                    found_col = col
                    break
            
            if found_col:
                info[f'{key}_dist'] = customer_data[found_col].value_counts().to_dict()
                print(f"  ✅ {key} 분포: {dict(list(info[f'{key}_dist'].items())[:3])}...")  # 상위 3개만 출력
            else:
                info[f'{key}_dist'] = {}
                print(f"  ⚠️ {key} 컬럼 없음")
        
        print(f"✅ 고객수: {info['total_customers']:,}명")
        return info
    
    def _check_lp_data_quality(self, lp_data):
        """LP 데이터 품질 점검"""
        # 데이터 타입 변환
        lp_data['LP수신일자'] = pd.to_datetime(lp_data['LP수신일자'])
        
        quality = {
            'total_records': len(lp_data),
            'date_range': {
                'start': lp_data['LP수신일자'].min(),
                'end': lp_data['LP수신일자'].max()
            },
            'unique_customers': lp_data['대체고객번호'].nunique(),
            'missing_values': lp_data.isnull().sum().to_dict(),
            'negative_values': (lp_data['순방향유효전력'] < 0).sum(),
            'zero_values': (lp_data['순방향유효전력'] == 0).sum(),
        }
        
        print(f"✅ 총 레코드: {quality['total_records']:,}건")
        print(f"✅ 기간: {quality['date_range']['start']} ~ {quality['date_range']['end']}")
        print(f"✅ 고객수: {quality['unique_customers']:,}명")
        print(f"✅ 음수값: {quality['negative_values']:,}건")
        print(f"✅ 0값: {quality['zero_values']:,}건")
        
        return quality
    
    def _calculate_completeness(self, lp_data):
        """데이터 완정성 계산"""
        lp_data['date'] = lp_data['LP수신일자'].dt.date
        lp_data['hour'] = lp_data['LP수신일자'].dt.hour
        lp_data['quarter_hour'] = (lp_data['LP수신일자'].dt.minute // 15) * 15
        
        # 15분 간격 정확성 체크
        expected_intervals = pd.date_range(
            start=lp_data['LP수신일자'].min(),
            end=lp_data['LP수신일자'].max(),
            freq='15min'
        )
        
        completeness = {
            'expected_records': len(expected_intervals) * lp_data['대체고객번호'].nunique(),
            'actual_records': len(lp_data),
            'completeness_rate': len(lp_data) / (len(expected_intervals) * lp_data['대체고객번호'].nunique()) * 100
        }
        
        # 고객별 완정성
        customer_completeness = lp_data.groupby('대체고객번호').size() / len(expected_intervals) * 100
        completeness['customer_completeness'] = {
            'mean': customer_completeness.mean(),
            'min': customer_completeness.min(),
            'max': customer_completeness.max(),
            'std': customer_completeness.std()
        }
        
        print(f"✅ 데이터 완정성: {completeness['completeness_rate']:.2f}%")
        
        return completeness
    
    def _detect_data_anomalies(self, lp_data):
        """데이터 이상치 탐지"""
        # 통계적 이상치 탐지
        Q1 = lp_data['순방향유효전력'].quantile(0.25)
        Q3 = lp_data['순방향유효전력'].quantile(0.75)
        IQR = Q3 - Q1
        
        outliers_iqr = lp_data[
            (lp_data['순방향유효전력'] < Q1 - 1.5 * IQR) | 
            (lp_data['순방향유효전력'] > Q3 + 1.5 * IQR)
        ]
        
        # Z-score 이상치
        z_scores = np.abs((lp_data['순방향유효전력'] - lp_data['순방향유효전력'].mean()) / lp_data['순방향유효전력'].std())
        outliers_zscore = lp_data[z_scores > 3]
        
        anomalies = {
            'iqr_outliers': len(outliers_iqr),
            'zscore_outliers': len(outliers_zscore),
            'outlier_rate_iqr': len(outliers_iqr) / len(lp_data) * 100,
            'outlier_rate_zscore': len(outliers_zscore) / len(lp_data) * 100
        }
        
        print(f"✅ IQR 이상치: {anomalies['outlier_rate_iqr']:.3f}%")
        print(f"✅ Z-score 이상치: {anomalies['outlier_rate_zscore']:.3f}%")
        
        return anomalies
    
    def _print_quality_summary(self):
        """품질 점검 요약 출력"""
        print("\n" + "="*50)
        print("📊 데이터 품질 점검 완료")
        print("="*50)
    
    # ============ 2단계: 기본 패턴 탐색 (60분) ============
    
    def analyze_basic_patterns(self, lp_data, customer_data=None, calendar_data=None):
        """
        기본 전력 사용 패턴 분석
        """
        print("\n📊 2단계: 기본 패턴 탐색 시작...")
        
        # 데이터 전처리
        processed_data = self._preprocess_for_pattern_analysis(lp_data, customer_data, calendar_data)
        
        # 시간별 패턴 분석
        time_patterns = self._analyze_time_patterns(processed_data)
        
        # 고객 세분화 기초 분석
        customer_segmentation = self._analyze_customer_segmentation(processed_data)
        
        self.pattern_analysis = {
            'time_patterns': time_patterns,
            'customer_segmentation': customer_segmentation,
            'processed_data': processed_data
        }
        
        return self.pattern_analysis
    
    def _preprocess_for_pattern_analysis(self, lp_data, customer_data, calendar_data):
        """패턴 분석을 위한 데이터 전처리"""
        print("  🔄 데이터 전처리 중...")
        
        # 시간 변수 생성
        lp_data['datetime'] = pd.to_datetime(lp_data['LP수신일자'])
        lp_data['date'] = lp_data['datetime'].dt.date
        lp_data['hour'] = lp_data['datetime'].dt.hour
        lp_data['weekday'] = lp_data['datetime'].dt.weekday  # 0=월요일
        lp_data['month'] = lp_data['datetime'].dt.month
        lp_data['quarter'] = lp_data['datetime'].dt.quarter
        lp_data['is_weekend'] = lp_data['weekday'].isin([5, 6])  # 토, 일
        
        # 일간 집계 데이터 생성
        daily_agg = lp_data.groupby(['대체고객번호', 'date']).agg({
            '순방향유효전력': ['sum', 'mean', 'max', 'min', 'std']
        }).reset_index()
        
        # 컬럼명 정리
        daily_agg.columns = ['customer_id', 'date', 'daily_sum', 'daily_mean', 'daily_max', 'daily_min', 'daily_std']
        
        # 날짜 관련 피처 다시 생성 (일간 집계 후)
        daily_agg['date_dt'] = pd.to_datetime(daily_agg['date'])
        daily_agg['weekday'] = daily_agg['date_dt'].dt.weekday  # 0=월요일
        daily_agg['month'] = daily_agg['date_dt'].dt.month
        daily_agg['quarter'] = daily_agg['date_dt'].dt.quarter
        daily_agg['is_weekend'] = daily_agg['weekday'].isin([5, 6])  # 토, 일
        
        print(f"  ✅ 시간 피처 생성 완료: weekday, month, quarter, is_weekend")
        
        # 고객 정보 병합
        if customer_data is not None:
            # 고객 데이터의 실제 컬럼명 확인
            customer_key_col = None
            possible_keys = ['대체고객번호', '고객번호', 'customer_id', 'Customer_ID']
            
            for col in possible_keys:
                if col in customer_data.columns:
                    customer_key_col = col
                    break
            
            if customer_key_col:
                daily_agg = daily_agg.merge(customer_data, left_on='customer_id', right_on=customer_key_col, how='left')
                print(f"  ✅ 고객 정보 병합 완료 (키: {customer_key_col})")
            else:
                print(f"  ⚠️ 고객 데이터 병합 실패 - 키 컬럼을 찾을 수 없음. 사용 가능한 컬럼: {list(customer_data.columns)}")
                print(f"  ℹ️ 고객 정보 없이 분석 계속...")
        
        # 기상/달력 정보 병합
        if calendar_data is not None:
            # 날짜 컬럼 확인 및 변환
            date_col = None
            possible_date_cols = ['date', '날짜', 'Date', 'DATE']
            
            for col in possible_date_cols:
                if col in calendar_data.columns:
                    date_col = col
                    break
            
            if date_col:
                # 기상 데이터가 weather_daily_processed.csv인 경우 날짜 형식 확인
                if 'year' in calendar_data.columns and 'month' in calendar_data.columns and 'day' in calendar_data.columns:
                    # year, month, day 컬럼으로 날짜 생성
                    calendar_data['date_parsed'] = pd.to_datetime(calendar_data[['year', 'month', 'day']])
                    calendar_data['date_for_merge'] = calendar_data['date_parsed'].dt.date
                    daily_agg = daily_agg.merge(calendar_data, left_on='date', right_on='date_for_merge', how='left')
                    print(f"  ✅ 기상/달력 정보 병합 완료 (year-month-day 기준)")
                else:
                    # 일반적인 date 컬럼 사용
                    calendar_data[date_col] = pd.to_datetime(calendar_data[date_col]).dt.date
                    daily_agg = daily_agg.merge(calendar_data, left_on='date', right_on=date_col, how='left')
                    print(f"  ✅ 기상/달력 정보 병합 완료 (키: {date_col})")
            else:
                print(f"  ⚠️ 기상/달력 데이터 병합 실패 - 날짜 컬럼을 찾을 수 없음. 사용 가능한 컬럼: {list(calendar_data.columns)}")
                print(f"  ℹ️ 기상/달력 정보 없이 분석 계속...")
        
        print(f"  ✅ 일간 집계 데이터: {len(daily_agg):,}건")
        return daily_agg
    
    def _analyze_time_patterns(self, data):
        """시간별 패턴 분석"""
        print("  📈 시간별 패턴 분석 중...")
        
        patterns = {}
        
        # 1. 시간대별 패턴 (일간 집계 데이터에서는 의미 없으므로 건너뛰기)
        # hourly_pattern = data.groupby('hour')['daily_mean'].agg(['mean', 'std', 'count'])
        # patterns['hourly'] = hourly_pattern
        
        # 2. 요일별 패턴
        if 'weekday' in data.columns:
            weekday_pattern = data.groupby('weekday')['daily_mean'].agg(['mean', 'std', 'count'])
            patterns['weekday'] = weekday_pattern
        
        # 3. 월별 패턴 (계절성)
        if 'month' in data.columns:
            monthly_pattern = data.groupby('month')['daily_mean'].agg(['mean', 'std', 'count'])
            patterns['monthly'] = monthly_pattern
        
        # 4. 주중/주말 패턴
        if 'is_weekend' in data.columns:
            weekend_pattern = data.groupby('is_weekend')['daily_mean'].agg(['mean', 'std', 'count'])
            patterns['weekend'] = weekend_pattern
        
        # 5. 업종별 패턴 (고객 데이터가 있는 경우)
        usage_purpose_col = None
        possible_usage_cols = ['사용용도', 'usage_purpose', 'Usage_Purpose']
        
        for col in possible_usage_cols:
            if col in data.columns:
                usage_purpose_col = col
                break
        
        if usage_purpose_col:
            industry_pattern = data.groupby(usage_purpose_col)['daily_mean'].agg(['mean', 'std', 'count'])
            patterns['industry'] = industry_pattern
            print(f"  ✅ {usage_purpose_col} 기준 업종별 패턴 분석 완료")
        
        print(f"  ✅ 시간별 패턴 분석 완료 ({len(patterns)}개 패턴)")
        return patterns
    
    def _analyze_customer_segmentation(self, data):
        """고객 세분화 기초 분석"""
        print("  👥 고객 세분화 분석 중...")
        
        # 고객별 평균 사용량 계산
        customer_avg = data.groupby('customer_id')['daily_mean'].mean()
        
        # 사용량 규모별 분류
        segmentation = {
            'large_users': customer_avg.quantile(0.9),  # 상위 10%
            'medium_users': customer_avg.quantile(0.5),  # 중간 50%
            'small_users': customer_avg.quantile(0.1),   # 하위 10%
        }
        
        # 고객별 사용량 분포
        customer_stats = {
            'customer_count': len(customer_avg),
            'usage_distribution': {
                'mean': customer_avg.mean(),
                'std': customer_avg.std(),
                'min': customer_avg.min(),
                'max': customer_avg.max(),
                'q25': customer_avg.quantile(0.25),
                'q50': customer_avg.quantile(0.50),
                'q75': customer_avg.quantile(0.75)
            },
            'segmentation_thresholds': segmentation
        }
        
        print(f"  ✅ {customer_stats['customer_count']:,}명 고객 세분화 완료")
        return customer_stats
    
    # ============ 3단계: 변동성 기초 분석 (90분) ============
    
    def analyze_variability(self, processed_data):
        """
        변동성 기초 분석 - 변동계수 설계를 위한 기초 작업
        """
        print("\n📈 3단계: 변동성 기초 분석 시작...")
        
        # 기본 변동성 지표 계산
        basic_variability = self._calculate_basic_variability(processed_data)
        
        # 변동성 패턴 분석
        variability_patterns = self._analyze_variability_patterns(processed_data)
        
        self.variability_analysis = {
            'basic_variability': basic_variability,
            'variability_patterns': variability_patterns
        }
        
        return self.variability_analysis
    
    def _calculate_basic_variability(self, data):
        """기본 변동성 지표 계산"""
        print("  📊 기본 변동성 지표 계산 중...")
        
        variability_metrics = {}
        
        # 고객별 변동계수 계산
        customer_cv = data.groupby('customer_id').apply(
            lambda x: x['daily_mean'].std() / x['daily_mean'].mean() if x['daily_mean'].mean() > 0 else np.nan
        )
        
        # 1. 일간 변동계수
        variability_metrics['daily_cv'] = {
            'mean': customer_cv.mean(),
            'std': customer_cv.std(),
            'distribution': customer_cv.describe()
        }
        
        # 2. 주간 변동계수 (주별 패턴의 일관성)
        try:
            # 주 번호 생성
            data_with_week = data.copy()
            data_with_week['week'] = pd.to_datetime(data_with_week['date']).dt.isocalendar().week
            
            weekly_cv = data_with_week.groupby(['customer_id', 'week']).agg({
                'daily_mean': ['mean', 'std']
            }).reset_index()
            weekly_cv.columns = ['customer_id', 'week', 'weekly_mean', 'weekly_std']
            weekly_cv['weekly_cv'] = weekly_cv['weekly_std'] / weekly_cv['weekly_mean']
            
            customer_weekly_cv = weekly_cv.groupby('customer_id')['weekly_cv'].mean()
            variability_metrics['weekly_cv'] = {
                'mean': customer_weekly_cv.mean(),
                'std': customer_weekly_cv.std(),
                'distribution': customer_weekly_cv.describe()
            }
        except Exception as e:
            print(f"    ⚠️ 주간 변동계수 계산 실패: {e}")
            variability_metrics['weekly_cv'] = {'mean': np.nan, 'std': np.nan}
        
        # 3. 월간 변동계수
        try:
            if 'month' in data.columns:
                monthly_cv = data.groupby(['customer_id', 'month']).agg({
                    'daily_mean': ['mean', 'std']
                }).reset_index()
                monthly_cv.columns = ['customer_id', 'month', 'monthly_mean', 'monthly_std']
                monthly_cv['monthly_cv'] = monthly_cv['monthly_std'] / monthly_cv['monthly_mean']
                
                customer_monthly_cv = monthly_cv.groupby('customer_id')['monthly_cv'].mean()
                variability_metrics['monthly_cv'] = {
                    'mean': customer_monthly_cv.mean(),
                    'std': customer_monthly_cv.std(),
                    'distribution': customer_monthly_cv.describe()
                }
            else:
                print("    ⚠️ month 컬럼이 없어 월간 변동계수 계산 건너뛰기")
                variability_metrics['monthly_cv'] = {'mean': np.nan, 'std': np.nan}
        except Exception as e:
            print(f"    ⚠️ 월간 변동계수 계산 실패: {e}")
            variability_metrics['monthly_cv'] = {'mean': np.nan, 'std': np.nan}
        
        # 4. 추가 변동성 지표
        # 범위 기반 변동성
        try:
            customer_range_cv = data.groupby('customer_id').apply(
                lambda x: (x['daily_mean'].max() - x['daily_mean'].min()) / x['daily_mean'].mean() if x['daily_mean'].mean() > 0 else np.nan
            )
            
            variability_metrics['range_based_cv'] = {
                'mean': customer_range_cv.mean(),
                'std': customer_range_cv.std(),
                'distribution': customer_range_cv.describe()
            }
        except Exception as e:
            print(f"    ⚠️ 범위 기반 변동계수 계산 실패: {e}")
            variability_metrics['range_based_cv'] = {'mean': np.nan, 'std': np.nan}
        
        print(f"  ✅ 기본 변동성 지표 계산 완료")
        return variability_metrics
    
    def _analyze_variability_patterns(self, data):
        """변동성 패턴 분석"""
        print("  🔍 변동성 패턴 분석 중...")
        
        patterns = {}
        
        # 1. 업종별 변동성 비교
        usage_purpose_col = None
        possible_usage_cols = ['사용용도', 'usage_purpose', 'Usage_Purpose']
        
        for col in possible_usage_cols:
            if col in data.columns:
                usage_purpose_col = col
                break
        
        if usage_purpose_col:
            try:
                industry_variability = data.groupby([usage_purpose_col, 'customer_id']).apply(
                    lambda x: x['daily_mean'].std() / x['daily_mean'].mean() if x['daily_mean'].mean() > 0 else np.nan
                ).reset_index()
                industry_variability.columns = [usage_purpose_col, 'customer_id', 'cv']
                
                industry_cv_summary = industry_variability.groupby(usage_purpose_col)['cv'].agg(['mean', 'std', 'count'])
                patterns['industry_variability'] = industry_cv_summary
                print(f"  ✅ {usage_purpose_col} 기준 업종별 변동성 분석 완료")
            except Exception as e:
                print(f"  ⚠️ 업종별 변동성 분석 실패: {e}")
        
        # 2. 계절별 변동성 차이
        try:
            if 'month' in data.columns:
                seasonal_variability = data.groupby(['customer_id', 'month']).apply(
                    lambda x: x['daily_mean'].std() / x['daily_mean'].mean() if x['daily_mean'].mean() > 0 else np.nan
                ).reset_index()
                seasonal_variability.columns = ['customer_id', 'month', 'cv']
                
                seasonal_cv_summary = seasonal_variability.groupby('month')['cv'].agg(['mean', 'std', 'count'])
                patterns['seasonal_variability'] = seasonal_cv_summary
                print(f"  ✅ 계절별 변동성 분석 완료")
            else:
                print("  ⚠️ month 컬럼이 없어 계절별 변동성 분석 건너뛰기")
        except Exception as e:
            print(f"  ⚠️ 계절별 변동성 분석 실패: {e}")
        
        # 3. 사용량 규모별 변동성
        try:
            customer_avg_usage = data.groupby('customer_id')['daily_mean'].mean()
            customer_cv = data.groupby('customer_id').apply(
                lambda x: x['daily_mean'].std() / x['daily_mean'].mean() if x['daily_mean'].mean() > 0 else np.nan
            )
            
            # 사용량 규모별 그룹핑
            usage_quantiles = customer_avg_usage.quantile([0.33, 0.67])
            def categorize_usage(usage):
                if usage <= usage_quantiles.iloc[0]:
                    return 'Low'
                elif usage <= usage_quantiles.iloc[1]:
                    return 'Medium'
                else:
                    return 'High'
            
            customer_usage_category = customer_avg_usage.apply(categorize_usage)
            usage_cv_df = pd.DataFrame({
                'usage_category': customer_usage_category,
                'cv': customer_cv
            })
            
            usage_cv_summary = usage_cv_df.groupby('usage_category')['cv'].agg(['mean', 'std', 'count'])
            patterns['usage_level_variability'] = usage_cv_summary
            print(f"  ✅ 사용량 규모별 변동성 분석 완료")
        except Exception as e:
            print(f"  ⚠️ 사용량 규모별 변동성 분석 실패: {e}")
        
        print(f"  ✅ 변동성 패턴 분석 완료 ({len(patterns)}개 패턴)")
        return patterns
    
    # ============ 4단계: 이상 패턴 탐지 (60분) ============
    
    def detect_anomalous_patterns(self, processed_data):
        """
        이상 패턴 탐지
        """
        print("\n🎯 4단계: 이상 패턴 탐지 시작...")
        
        # 통계적 이상치 식별
        statistical_outliers = self._identify_statistical_outliers(processed_data)
        
        # 시계열 이상치 탐지
        temporal_anomalies = self._detect_temporal_anomalies(processed_data)
        
        # 비정상 패턴 정의
        abnormal_patterns = self._define_abnormal_patterns(processed_data)
        
        anomaly_results = {
            'statistical_outliers': statistical_outliers,
            'temporal_anomalies': temporal_anomalies,
            'abnormal_patterns': abnormal_patterns
        }
        
        return anomaly_results
    
    def _identify_statistical_outliers(self, data):
        """통계적 이상치 식별"""
        print("  🔍 통계적 이상치 식별 중...")
        
        outliers = {}
        
        # IQR 방법
        Q1 = data['daily_mean'].quantile(0.25)
        Q3 = data['daily_mean'].quantile(0.75)
        IQR = Q3 - Q1
        
        iqr_outliers = data[
            (data['daily_mean'] < Q1 - 1.5 * IQR) | 
            (data['daily_mean'] > Q3 + 1.5 * IQR)
        ]
        
        outliers['iqr_outliers'] = {
            'count': len(iqr_outliers),
            'rate': len(iqr_outliers) / len(data) * 100,
            'customer_count': iqr_outliers['customer_id'].nunique()
        }
        
        # Z-score 방법
        z_scores = np.abs((data['daily_mean'] - data['daily_mean'].mean()) / data['daily_mean'].std())
        zscore_outliers = data[z_scores > 3]
        
        outliers['zscore_outliers'] = {
            'count': len(zscore_outliers),
            'rate': len(zscore_outliers) / len(data) * 100,
            'customer_count': zscore_outliers['customer_id'].nunique()
        }
        
        print(f"  ✅ IQR 이상치: {outliers['iqr_outliers']['count']:,}건 ({outliers['iqr_outliers']['rate']:.2f}%)")
        print(f"  ✅ Z-score 이상치: {outliers['zscore_outliers']['count']:,}건 ({outliers['zscore_outliers']['rate']:.2f}%)")
        
        return outliers
    
    def _detect_temporal_anomalies(self, data):
        """시계열 이상치 탐지"""
        print("  ⏰ 시계열 이상치 탐지 중...")
        
        temporal_anomalies = {}
        
        # 고객별 시계열 이상치 탐지
        for customer_id in data['customer_id'].unique()[:100]:  # 샘플로 100명만
            customer_data = data[data['customer_id'] == customer_id].sort_values('date')
            
            if len(customer_data) < 30:  # 최소 30일 데이터 필요
                continue
            
            # 급격한 증가/감소 탐지 (>200% 변화)
            customer_data['pct_change'] = customer_data['daily_mean'].pct_change()
            sudden_changes = customer_data[abs(customer_data['pct_change']) > 2.0]  # 200% 변화
            
            # 연속적인 0값 탐지
            zero_streaks = customer_data[customer_data['daily_mean'] == 0]
            
            if len(sudden_changes) > 0 or len(zero_streaks) > 5:  # 5일 이상 연속 0값
                temporal_anomalies[customer_id] = {
                    'sudden_changes': len(sudden_changes),
                    'zero_streaks': len(zero_streaks)
                }
        
        print(f"  ✅ {len(temporal_anomalies):,}명 고객에서 시계열 이상 탐지")
        
        return temporal_anomalies
    
    def _define_abnormal_patterns(self, data):
        """비정상 패턴 정의"""
        print("  📋 비정상 패턴 정의 중...")
        
        abnormal_patterns = {
            'pattern_definitions': {
                1: '전력 사용 급증/급감 (사업 확장/축소)',
                2: '사용 패턴 변화 (운영시간 변경)', 
                3: '효율성 급변 (설비 교체/고장)',
                4: '계절성 이탈 (사업 모델 변화)'
            },
            'detection_criteria': {
                'usage_spike': 'daily_mean > mean + 3*std',
                'usage_drop': 'daily_mean < mean - 3*std',
                'pattern_shift': 'monthly pattern change > 50%',
                'efficiency_change': 'weekly efficiency variance > threshold'
            }
        }
        
        print(f"  ✅ {len(abnormal_patterns['pattern_definitions'])}가지 비정상 패턴 정의 완료")
        
        return abnormal_patterns
    
    # ============ 5단계: 전처리 방향 결정 (30분) ============
    
    def decide_preprocessing_strategy(self, data_quality_report, pattern_analysis, variability_analysis):
        """
        전처리 방향 결정
        """
        print("\n🔧 5단계: 전처리 방향 결정...")
        
        preprocessing_strategy = {
            'missing_data_handling': self._decide_missing_data_strategy(data_quality_report),
            'outlier_handling': self._decide_outlier_strategy(data_quality_report),
            'normalization_method': self._decide_normalization_strategy(pattern_analysis),
            'feature_engineering': self._decide_feature_engineering(pattern_analysis, variability_analysis)
        }
        
        self._print_preprocessing_summary(preprocessing_strategy)
        
        return preprocessing_strategy
    
    def _decide_missing_data_strategy(self, quality_report):
        """결측치 처리 전략 결정"""
        completeness_rate = quality_report['data_completeness']['completeness_rate']
        
        if completeness_rate > 95:
            strategy = "선형보간 또는 forward fill"
        elif completeness_rate > 80:
            strategy = "계절성 고려 보간"
        else:
            strategy = "장기 결측 기간 분석 제외"
        
        return {
            'completeness_rate': completeness_rate,
            'recommended_strategy': strategy
        }
    
    def _decide_outlier_strategy(self, quality_report):
        """이상치 처리 전략 결정"""
        outlier_rate = quality_report['anomaly_detection']['outlier_rate_iqr']
        
        if outlier_rate < 1:
            strategy = "이상치 유지 (정상 범위)"
        elif outlier_rate < 5:
            strategy = "extreme outlier만 제거"
        else:
            strategy = "robust 통계량 사용"
        
        return {
            'outlier_rate': outlier_rate,
            'recommended_strategy': strategy
        }
    
    def _decide_normalization_strategy(self, pattern_analysis):
        """정규화 방법 결정"""
        customer_stats = pattern_analysis['customer_segmentation']
        usage_std = customer_stats['usage_distribution']['std']
        usage_mean = customer_stats['usage_distribution']['mean']
        cv = usage_std / usage_mean if usage_mean > 0 else 0
        
        if cv > 1.0:
            strategy = "고객별 표준화 + 로그 변환"
        elif cv > 0.5:
            strategy = "고객별 표준화"
        else:
            strategy = "전체 Min-Max 정규화"
        
        return {
            'coefficient_of_variation': cv,
            'recommended_strategy': strategy
        }
    
    def _decide_feature_engineering(self, pattern_analysis, variability_analysis):
        """피처 엔지니어링 전략 결정"""
        features_to_create = []
        
        # 시간 기반 피처
        time_patterns = pattern_analysis['time_patterns']
        if 'monthly' in time_patterns:
            features_to_create.extend([
                'month_sin', 'month_cos',  # 계절 순환 피처
                'is_summer', 'is_winter',  # 계절 더미 변수
            ])
        
        # 요일 기반 피처
        if 'weekday' in time_patterns:
            features_to_create.extend([
                'weekday_sin', 'weekday_cos',  # 요일 순환 피처
                'is_weekend'                   # 주말 여부
            ])
        
        # 변동성 기반 피처
        if variability_analysis:
            features_to_create.extend([
                'rolling_mean_7d',       # 7일 이동평균
                'rolling_std_7d',        # 7일 이동표준편차
                'usage_volatility',      # 변동성 지수
            ])
        
        # 고객 기반 피처
        features_to_create.extend([
            'customer_avg_usage',      # 고객 평균 사용량
            'customer_usage_rank',     # 고객 사용량 순위
            'deviation_from_avg'       # 평균 대비 편차
        ])
        
        return {
            'features_to_create': features_to_create,
            'total_features': len(features_to_create)
        }
    
    def _print_preprocessing_summary(self, strategy):
        """전처리 전략 요약 출력"""
        print("\n" + "="*60)
        print("🔧 전처리 전략 결정 완료")
        print("="*60)
        
        print(f"📋 결측치 처리: {strategy['missing_data_handling']['recommended_strategy']}")
        print(f"🎯 이상치 처리: {strategy['outlier_handling']['recommended_strategy']}")
        print(f"📊 정규화 방법: {strategy['normalization_method']['recommended_strategy']}")
        print(f"🛠️ 생성할 피처: {strategy['feature_engineering']['total_features']}개")
        
        print("\n💡 변동계수 설계를 위한 인사이트:")
        print("- 어떤 변동성 지표가 실제 사업 변화를 잘 반영하는가?")
        print("- 업종별로 다른 임계값이 필요한가?")
        print("- 시간 윈도우는 얼마나 설정해야 하는가?")
        print("- 계절성 보정이 필요한가?")
    
    # ============ 시각화 및 리포트 생성 ============
    
    def create_eda_visualizations(self, processed_data):
        """탐색적 데이터 분석 시각화"""
        print("\n📈 EDA 시각화 생성 중...")
        
        # 1. 요일별 사용 패턴 (시간대별 대신)
        self._plot_weekday_patterns_daily(processed_data)
        
        # 2. 요일별 사용 패턴 (바 차트)
        self._plot_weekday_patterns(processed_data)
        
        # 3. 월별 사용량 박스플롯
        self._plot_monthly_boxplot(processed_data)
        
        # 4. 고객별 사용량 분포
        self._plot_customer_distribution(processed_data)
        
        print("✅ 시각화 생성 완료")
    
    def _plot_weekday_patterns_daily(self, data):
        """요일별 평균 사용 패턴 (라인 차트)"""
        if 'weekday' in data.columns:
            weekday_avg = data.groupby('weekday')['daily_mean'].mean()
            
            plt.figure(figsize=(10, 6))
            plt.plot(range(7), weekday_avg.values, marker='o', linewidth=2, markersize=8)
            plt.title('Daily Power Usage by Weekday', fontsize=14, fontweight='bold')
            plt.xlabel('Weekday')
            plt.ylabel('Average Usage (kWh)')
            plt.xticks(range(7), ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
        else:
            print("  ⚠️ 요일별 패턴 시각화 불가 - weekday 컬럼 없음")
    
    def _plot_weekday_patterns(self, data):
        """요일별 사용 패턴"""
        if 'weekday' not in data.columns:
            print("  ⚠️ 요일별 패턴 시각화 불가 - weekday 컬럼 없음")
            return
            
        weekday_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        weekday_avg = data.groupby('weekday')['daily_mean'].mean()
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(range(7), weekday_avg.values, color=['skyblue' if i < 5 else 'lightcoral' for i in range(7)])
        plt.title('Average Power Usage by Weekday', fontsize=14, fontweight='bold')
        plt.xlabel('Weekday')
        plt.ylabel('Average Usage (kWh)')
        plt.xticks(range(7), weekday_names)
        plt.grid(True, alpha=0.3, axis='y')
        
        # 주중/주말 구분 표시
        for i, bar in enumerate(bars):
            if i >= 5:  # 주말
                bar.set_label('Weekend' if i == 5 else '')
            else:  # 주중
                bar.set_label('Weekday' if i == 0 else '')
        
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    def _plot_monthly_boxplot(self, data):
        """월별 사용량 박스플롯"""
        if 'month' not in data.columns:
            print("  ⚠️ 월별 패턴 시각화 불가 - month 컬럼 없음")
            return
            
        plt.figure(figsize=(14, 8))
        
        # 월별 데이터 준비
        monthly_data = [data[data['month'] == m]['daily_mean'].values for m in range(1, 13)]
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        # 데이터가 있는 월만 표시
        valid_months = []
        valid_data = []
        valid_names = []
        
        for i, month_data in enumerate(monthly_data):
            if len(month_data) > 0:
                valid_months.append(i + 1)
                valid_data.append(month_data)
                valid_names.append(month_names[i])
        
        if not valid_data:
            print("  ⚠️ 월별 데이터 없음")
            return
        
        box_plot = plt.boxplot(valid_data, labels=valid_names, patch_artist=True)
        
        # 계절별 색상 구분
        colors = []
        for month in valid_months:
            if month in [12, 1, 2]:  # 겨울
                colors.append('lightblue')
            elif month in [3, 4, 5]:  # 봄
                colors.append('lightgreen')
            elif month in [6, 7, 8]:  # 여름
                colors.append('lightcoral')
            else:  # 가을
                colors.append('orange')
        
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        plt.title('Monthly Power Usage Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('Month')
        plt.ylabel('Daily Average Usage (kWh)')
        plt.grid(True, alpha=0.3, axis='y')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    def _plot_customer_distribution(self, data):
        """고객별 평균 사용량 분포"""
        customer_avg = data.groupby('customer_id')['daily_mean'].mean()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 히스토그램
        ax1.hist(customer_avg.values, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_title('Customer Average Usage Distribution', fontweight='bold')
        ax1.set_xlabel('Average Usage (kWh)')
        ax1.set_ylabel('Number of Customers')
        ax1.grid(True, alpha=0.3)
        
        # 박스플롯
        ax2.boxplot(customer_avg.values, vert=True)
        ax2.set_title('Customer Average Usage Boxplot', fontweight='bold')
        ax2.set_ylabel('Average Usage (kWh)')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def generate_eda_report(self):
        """EDA 종합 리포트 생성"""
        print("\n" + "="*70)
        print("KEPCO LP Data Preprocessing and EDA Report")
        print("="*70)
        
        if hasattr(self, 'data_quality_report'):
            print("\n🔍 Step 1: Data Quality Check Results")
            print("-" * 40)
            quality = self.data_quality_report
            print(f"Total Records: {quality['lp_quality']['total_records']:,}")
            print(f"Customers: {quality['lp_quality']['unique_customers']:,}")
            print(f"Data Completeness: {quality['data_completeness']['completeness_rate']:.2f}%")
            print(f"Outlier Rate: {quality['anomaly_detection']['outlier_rate_iqr']:.3f}%")
        
        if hasattr(self, 'pattern_analysis'):
            print("\n📊 Step 2: Basic Pattern Analysis Results")
            print("-" * 40)
            pattern = self.pattern_analysis
            if 'customer_segmentation' in pattern:
                seg = pattern['customer_segmentation']
                print(f"Analyzed Customers: {seg['customer_count']:,}")
                print(f"Average Usage: {seg['usage_distribution']['mean']:.2f} kWh")
                print(f"Usage Std Dev: {seg['usage_distribution']['std']:.2f} kWh")
        
        if hasattr(self, 'variability_analysis'):
            print("\n📈 Step 3: Variability Analysis Results")
            print("-" * 40)
            var = self.variability_analysis
            if 'basic_variability' in var:
                basic = var['basic_variability']
                print(f"Average Daily CV: {basic['daily_cv']['mean']:.4f}")
                if not pd.isna(basic['weekly_cv']['mean']):
                    print(f"Average Weekly CV: {basic['weekly_cv']['mean']:.4f}")
                if not pd.isna(basic['monthly_cv']['mean']):
                    print(f"Average Monthly CV: {basic['monthly_cv']['mean']:.4f}")
        
        print("\n💡 Next Steps Recommendations:")
        print("- Define and design variability coefficient")
        print("- Implement stacking ensemble model") 
        print("- Apply overfitting prevention techniques")
        print("- Develop business activity change prediction algorithm")
        
        print("\n" + "="*70)

# 사용 예시 - LP 데이터 파일들을 결합하여 분석

# 데이터 로딩 (여러 LP 파일들 결합)
lp_files = ['LP데이터1.csv', 'LP데이터2.csv']  # 한 달을 반으로 나눈 파일들
customer_data = pd.read_excel('고객번호.xlsx')
weather_data = pd.read_csv('weather_daily_processed.csv') 
calendar_data = pd.read_csv('power_analysis_calendar_2022_2025.csv')

# 전처리 및 EDA 실행
preprocessor = KepcoDataPreprocessor()

# LP 데이터 결합
combined_lp_data = preprocessor.load_and_combine_lp_data(lp_files)

# 1단계: 데이터 품질 점검 (30분)
quality_report = preprocessor.check_data_quality(combined_lp_data, customer_data)

# 2단계: 기본 패턴 탐색 (60분) - 기상 데이터도 함께 병합!
pattern_analysis = preprocessor.analyze_basic_patterns(combined_lp_data, customer_data, weather_data)

# 3단계: 변동성 기초 분석 (90분)
variability_analysis = preprocessor.analyze_variability(pattern_analysis['processed_data'])

# 4단계: 이상 패턴 탐지 (60분)
anomaly_results = preprocessor.detect_anomalous_patterns(pattern_analysis['processed_data'])

# 5단계: 전처리 방향 결정 (30분)
preprocessing_strategy = preprocessor.decide_preprocessing_strategy(
    quality_report, pattern_analysis, variability_analysis
)

# 시각화 생성 (폰트 오류 없이)
preprocessor.create_eda_visualizations(pattern_analysis['processed_data'])

# 종합 리포트 생성
preprocessor.generate_eda_report()

print("✅ Complete LP data preprocessing and EDA finished!")
print("📤 Next: Define variability coefficient and implement stacking model")
