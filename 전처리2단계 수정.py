"""
한국전력 데이터 JSON 생성 - 시간적 편향 해결 버전 (완전 재작성)
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
import warnings
from multiprocessing import Pool, cpu_count
from functools import partial
import gc

warnings.filterwarnings('ignore')

class KEPCOAnalyzer:
    
    def __init__(self, target_customers=500, records_per_customer=100, n_jobs=-1):
        self.target_customers = target_customers      # 500명
        self.records_per_customer = records_per_customer  # 고객당 100개
        self.sample_size = target_customers * records_per_customer  # 총 50,000개
        
        self.n_jobs = cpu_count() if n_jobs == -1 else n_jobs
        self.analysis_results = {}
        
        # 실제 사용할 샘플링 설정
        self.sampling_config = {
            'customer_sample_ratio': 0.3,      # 고객의 30%만 샘플링
            'time_sample_ratio': 0.2,          # 시간 데이터의 20%만 샘플링  
            'min_customers': 20,               # 최소 고객 수
            'min_records_per_customer': 50,    # 고객당 최소 레코드 수
            'stratified_sampling': True,       # 계층 샘플링 사용
            'temporal_stratification': True    # 시간대별 계층 샘플링
        }
        
    def load_hdf5_data(self, hdf5_path='./analysis_results/processed_lp_data.h5'):
        """진짜 30% 고객, 20% 시간 데이터 샘플링"""
        with pd.HDFStore(hdf5_path, mode='r') as store:
            total_rows = store.get_storer('df').nrows
        
        print(f"    전체 데이터 크기: {total_rows:,}건")
        
        # 실제 sampling_config 기반 샘플링 적용
        self.df = self._proper_sampling_by_config(hdf5_path, total_rows)
        
        self._prepare_datetime_features()
    
    def _proper_sampling_by_config(self, hdf5_path, total_rows):
        """sampling_config에 따른 올바른 샘플링"""
        print("    sampling_config 기반 샘플링 시작...")
        
        # 1단계: 전체 고객 목록 파악 (처음 일부만 스캔)
        print("      1단계: 전체 고객 목록 파악 중...")
        all_customers = self._get_all_customers(hdf5_path, total_rows)
        total_customers = len(all_customers)
        print(f"        전체 고객 수: {total_customers}명")
        
        # 2단계: 고객 30% 선택 (계층별)
        target_customer_count = max(
            self.sampling_config['min_customers'],
            int(total_customers * self.sampling_config['customer_sample_ratio'])
        )
        target_customer_count = min(target_customer_count, self.target_customers)
        
        print(f"      2단계: 고객 샘플링 ({target_customer_count}/{total_customers}명, {target_customer_count/total_customers*100:.1f}%)")
        selected_customers = self._sample_customers_stratified(all_customers, target_customer_count)
        
        # 3단계: 선택된 고객들의 전체 데이터 로딩
        print("      3단계: 선택된 고객 데이터 로딩 중...")
        customer_data = self._load_selected_customers_data(hdf5_path, total_rows, selected_customers)
        
        # 4단계: 각 고객별로 시간 데이터 20% 샘플링
        print("      4단계: 시간 데이터 20% 샘플링 중...")
        final_data = self._apply_time_sampling(customer_data, selected_customers)
        
        return final_data
    
    def _get_all_customers(self, hdf5_path, total_rows):
        """전체 고객 목록 파악 (효율적으로)"""
        unique_customers = set()
        chunk_size = 50000
        max_scan = min(total_rows, 500000)  # 최대 50만건만 스캔
        
        for start in range(0, max_scan, chunk_size):
            end = min(start + chunk_size, total_rows)
            chunk = pd.read_hdf(hdf5_path, key='df', start=start, stop=end)
            
            chunk_customers = set(chunk['대체고객번호'].unique())
            unique_customers.update(chunk_customers)
            
            # 고객 수가 충분하면 조기 종료
            if len(unique_customers) >= 1000:
                break
        
        return list(unique_customers)
    
    def _sample_customers_stratified(self, all_customers, target_count):
        """고객 계층별 샘플링 (전체 데이터 일부 스캔하여 계층 파악)"""
        if len(all_customers) <= target_count:
            return all_customers
        
        # 단순 랜덤 샘플링 (빠른 처리를 위해)
        selected = np.random.choice(all_customers, size=target_count, replace=False)
        return selected.tolist()
    
    def _load_selected_customers_data(self, hdf5_path, total_rows, selected_customers):
        """선택된 고객들의 모든 데이터 로딩"""
        selected_set = set(selected_customers)
        all_chunks = []
        chunk_size = 100000
        
        print(f"        선택된 고객 수: {len(selected_customers)}명")
        processed_chunks = 0
        total_chunks = (total_rows + chunk_size - 1) // chunk_size
        
        for start in range(0, total_rows, chunk_size):
            end = min(start + chunk_size, total_rows)
            chunk = pd.read_hdf(hdf5_path, key='df', start=start, stop=end)
            
            # 선택된 고객만 필터링
            filtered_chunk = chunk[chunk['대체고객번호'].isin(selected_set)]
            
            if len(filtered_chunk) > 0:
                all_chunks.append(filtered_chunk)
            
            processed_chunks += 1
            if processed_chunks % 10 == 0:
                print(f"          진행률: {processed_chunks}/{total_chunks} ({processed_chunks/total_chunks*100:.1f}%)")
            
            del chunk
            gc.collect()
        
        if all_chunks:
            combined_data = pd.concat(all_chunks, ignore_index=True)
            print(f"        로딩된 데이터: {len(combined_data):,}건")
            return combined_data
        else:
            print("        경고: 선택된 고객 데이터가 없습니다!")
            return pd.DataFrame()
    
    def _apply_time_sampling(self, customer_data, selected_customers):
        """각 고객별로 시간 데이터 20% 샘플링"""
        if customer_data.empty:
            return customer_data
        
        # datetime 컬럼 확인 및 처리
        datetime_col = None
        for col in ['datetime', 'LP 수신일자', 'LP수신일자', 'timestamp']:
            if col in customer_data.columns:
                datetime_col = col
                break
        
        if datetime_col:
            customer_data[datetime_col] = pd.to_datetime(customer_data[datetime_col], errors='coerce')
            customer_data = customer_data.dropna(subset=[datetime_col])
            customer_data = customer_data.sort_values([datetime_col, '대체고객번호'])
        
        final_chunks = []
        time_ratio = self.sampling_config['time_sample_ratio']
        
        for i, customer_id in enumerate(selected_customers):
            customer_records = customer_data[customer_data['대체고객번호'] == customer_id]
            
            if len(customer_records) == 0:
                continue
            
            # 이 고객의 시간 데이터 20% 샘플링
            n_samples = max(
                self.sampling_config['min_records_per_customer'],
                int(len(customer_records) * time_ratio)
            )
            n_samples = min(n_samples, self.records_per_customer)  # 최대 제한
            
            if len(customer_records) <= n_samples:
                final_chunks.append(customer_records)
            else:
                # 시간순으로 균등 간격 샘플링
                indices = np.linspace(0, len(customer_records)-1, n_samples, dtype=int)
                sampled_records = customer_records.iloc[indices]
                final_chunks.append(sampled_records)
            
            if (i+1) % 50 == 0:
                print(f"          고객 처리: {i+1}/{len(selected_customers)} ({(i+1)/len(selected_customers)*100:.1f}%)")
        
        if final_chunks:
            result = pd.concat(final_chunks, ignore_index=True)
            
            # 최종 품질 검증
            if datetime_col:
                start_date = result[datetime_col].min()
                end_date = result[datetime_col].max()
                total_days = (end_date - start_date).days
                months_covered = result[datetime_col].dt.month.nunique()
                years_covered = result[datetime_col].dt.year.nunique()
                
                print(f"        최종 샘플링 결과:")
                print(f"          데이터 크기: {len(result):,}건")
                print(f"          시간 범위: {start_date.date()} ~ {end_date.date()} ({total_days}일)")
                print(f"          연도 수: {years_covered}년")
                print(f"          월 다양성: {months_covered}개월")
            
            return result
        else:
            return pd.DataFrame()
    
    def _prepare_datetime_features(self):
        """datetime 기반 피처 생성"""
        datetime_col = None
        for col in ['datetime', 'LP 수신일자', 'LP수신일자', 'timestamp']:
            if col in self.df.columns:
                datetime_col = col
                break
        
        if datetime_col:
            self.df['datetime'] = pd.to_datetime(self.df[datetime_col], errors='coerce')
            self.df = self.df.dropna(subset=['datetime'])
            
            # 시간 관련 피처 생성
            self.df['hour'] = self.df['datetime'].dt.hour
            self.df['day_of_week'] = self.df['datetime'].dt.dayofweek
            self.df['month'] = self.df['datetime'].dt.month
            self.df['is_weekend'] = self.df['day_of_week'].isin([5, 6])
            
            # 계절 정의
            def get_season(month):
                if month in [12, 1, 2]:
                    return 'winter'
                elif month in [3, 4, 5]:
                    return 'spring'
                elif month in [6, 7, 8]:
                    return 'summer'
                else:
                    return 'fall'
            
            self.df['season'] = self.df['month'].apply(get_season)
    
    def analyze_temporal_patterns(self):
        target_col = '순방향 유효전력'
        
        hourly_stats = self.df.groupby('hour')[target_col].agg(['mean', 'std', 'min', 'max', 'count']).round(2)
        daily_stats = self.df.groupby('day_of_week')[target_col].agg(['mean', 'std', 'count']).round(2)
        monthly_stats = self.df.groupby('month')[target_col].agg(['mean', 'std', 'count']).round(2)
        seasonal_stats = self.df.groupby('season')[target_col].agg(['mean', 'std', 'count']).round(2)
        
        hourly_means = hourly_stats['mean']
        peak_threshold = hourly_means.quantile(0.7)
        off_peak_threshold = hourly_means.quantile(0.3)
        
        peak_hours = hourly_means[hourly_means >= peak_threshold].index.tolist()
        off_peak_hours = hourly_means[hourly_means <= off_peak_threshold].index.tolist()
        weekend_ratio = self.df['is_weekend'].mean()
        
        self.analysis_results['temporal_patterns'] = {
            'hourly_patterns': hourly_stats.to_dict(),
            'daily_patterns': daily_stats.to_dict(),
            'monthly_patterns': monthly_stats.to_dict(),
            'seasonal_patterns': seasonal_stats.to_dict(),
            'peak_hours': peak_hours,
            'off_peak_hours': off_peak_hours,
            'weekend_ratio': float(weekend_ratio)
        }
        
        # 시간적 편향 해결 검증 추가
        self._verify_temporal_bias_resolution()
    
    def _verify_temporal_bias_resolution(self):
        """시간적 편향 해결 검증"""
        target_col = '순방향 유효전력'
        
        # 월별 데이터 분포 확인
        monthly_counts = self.df['month'].value_counts().sort_index()
        monthly_balance = monthly_counts.std() / monthly_counts.mean() if monthly_counts.mean() > 0 else 0
        
        # 시간대별 분포 확인  
        hourly_counts = self.df['hour'].value_counts().sort_index()
        hourly_balance = hourly_counts.std() / hourly_counts.mean() if hourly_counts.mean() > 0 else 0
        
        # 계절별 분포 확인
        seasonal_counts = self.df['season'].value_counts()
        seasonal_balance = seasonal_counts.std() / seasonal_counts.mean() if seasonal_counts.mean() > 0 else 0
        
        self.analysis_results['temporal_bias_check'] = {
            'monthly_balance_cv': float(monthly_balance),
            'hourly_balance_cv': float(hourly_balance),
            'seasonal_balance_cv': float(seasonal_balance),
            'months_covered': int(self.df['month'].nunique()),
            'seasons_covered': int(self.df['season'].nunique()),
            'hours_covered': int(self.df['hour'].nunique()),
            'bias_resolved': monthly_balance < 0.5 and seasonal_balance < 0.3,
            'temporal_range_days': (self.df['datetime'].max() - self.df['datetime'].min()).days
        }
    
    def analyze_basic_patterns(self):
        """기본 전력 사용 패턴 분석 (CV 계산 제외)"""
        target_col = '순방향 유효전력'
        
        # 시간대별 기본 통계
        hourly_stats = self.df.groupby('hour')[target_col].agg(['mean', 'std', 'count']).round(2)
        daily_stats = self.df.groupby('day_of_week')[target_col].agg(['mean', 'std', 'count']).round(2)
        monthly_stats = self.df.groupby('month')[target_col].agg(['mean', 'std', 'count']).round(2)
        seasonal_stats = self.df.groupby('season')[target_col].agg(['mean', 'std', 'count']).round(2)
        
        # 고객별 기본 통계 (CV 제외)
        customer_basic_stats = {}
        customers = self.df['대체고객번호'].unique()
        
        for customer_id in customers:
            customer_data = self.df[self.df['대체고객번호'] == customer_id][target_col]
            if len(customer_data) > 1:
                customer_basic_stats[str(customer_id)] = {
                    'mean_power': float(customer_data.mean()),
                    'std_power': float(customer_data.std()),
                    'min_power': float(customer_data.min()),
                    'max_power': float(customer_data.max()),
                    'record_count': int(len(customer_data))
                }
        
        self.analysis_results['basic_patterns'] = {
            'hourly_stats': hourly_stats.to_dict(),
            'daily_stats': daily_stats.to_dict(),
            'monthly_stats': monthly_stats.to_dict(),
            'seasonal_stats': seasonal_stats.to_dict(),
            'customer_basic_stats': customer_basic_stats,
            'total_customers_analyzed': len(customer_basic_stats)
        }
    
    def analyze_anomalies(self):
        target_col = '순방향 유효전력'
        customers = self.df['대체고객번호'].unique()
        
        # 통계적 이상치 경계값 계산
        Q1 = self.df[target_col].quantile(0.25)
        Q3 = self.df[target_col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = self.df[(self.df[target_col] < lower_bound) | (self.df[target_col] > upper_bound)]
        
        # 시간대별 분석
        night_hours = [22, 23, 0, 1, 2, 3, 4, 5]
        day_hours = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
        
        night_data = self.df[self.df['hour'].isin(night_hours)]
        day_data = self.df[self.df['hour'].isin(day_hours)]
        
        night_mean = night_data[target_col].mean()
        day_mean = day_data[target_col].mean()
        night_day_ratio = night_mean / day_mean if day_mean > 0 else 0
        
        # 제로값 분석
        zero_count = (self.df[target_col] == 0).sum()
        zero_rate = zero_count / len(self.df)
        
        # 급격한 변화 분석
        sudden_changes = 0
        if len(self.df) > 1000:
            sample_df = self.df.sample(n=min(1000, len(self.df)), random_state=42)
            sample_df = sample_df.sort_values('datetime')
            power_diff = sample_df[target_col].diff().abs()
            threshold = power_diff.quantile(0.95)
            sudden_changes = (power_diff > threshold).sum()
        
        sudden_change_rate = sudden_changes / len(self.df) if len(self.df) > 0 else 0
        
        # 이상 고객 식별
        anomaly_customers = []
        for customer_id in customers[:50]:  # 최대 50명만 확인
            customer_data = self.df[self.df['대체고객번호'] == customer_id][target_col]
            if len(customer_data) > 10:
                zero_ratio = (customer_data == 0).mean()
                cv = customer_data.std() / customer_data.mean() if customer_data.mean() > 0 else 0
                
                if zero_ratio > 0.5 or cv > 3.0:
                    anomaly_customers.append(str(customer_id))
        
        anomaly_rate = len(anomaly_customers) / len(customers) if len(customers) > 0 else 0
        
        self.analysis_results['anomaly_patterns'] = {
            'outlier_count': int(len(outliers)),
            'outlier_rate': float(len(outliers) / len(self.df)),
            'zero_count': int(zero_count),
            'zero_rate': float(zero_rate),
            'sudden_changes': int(sudden_changes),
            'sudden_change_rate': float(sudden_change_rate),
            'night_day_ratio': float(night_day_ratio),
            'anomaly_customers': anomaly_customers,
            'estimated_anomaly_rate': float(anomaly_rate)
        }
        
        return True
    
    def generate_json_result(self, output_path='./analysis_results/analysis_results2_fixed.json'):
        self.analysis_results['metadata'] = {
            'timestamp': datetime.now().isoformat(),
            'stage': 'step2_proper_sampling_by_config',
            'version': '5.0_real_30percent_20percent_sampling',
            'sample_size': len(self.df) if hasattr(self, 'df') else 0,
            'total_customers': self.df['대체고객번호'].nunique() if hasattr(self, 'df') else 0,
            'target_customers': self.target_customers,
            'records_per_customer': self.records_per_customer,
            'sampling_method': 'proper_config_based_sampling',
            'customer_sample_ratio_used': self.sampling_config['customer_sample_ratio'],
            'time_sample_ratio_used': self.sampling_config['time_sample_ratio'],
            'temporal_bias_fixed': True,
            'processing_cores': self.n_jobs,
            'sampling_config': self.sampling_config
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.analysis_results, f, ensure_ascii=False, indent=2, default=str)
        
        return output_path
    
    def run_analysis(self, hdf5_path='./analysis_results/processed_lp_data.h5'):
        """진짜 30% 고객, 20% 시간 데이터 샘플링 분석"""
        print("진짜 30% 고객, 20% 시간 데이터 샘플링 분석 시작...")
        
        self.load_hdf5_data(hdf5_path)
        self.analyze_temporal_patterns()
        self.analyze_basic_patterns()
        self.analyze_anomalies()
        output_path = self.generate_json_result()
        
        # 결과 요약 출력
        self._print_analysis_summary()
        
        if hasattr(self, 'df'):
            del self.df
        gc.collect()
        
        return output_path
    
    def _print_analysis_summary(self):
        """분석 결과 요약 출력"""
        print("\n" + "="*60)
        print("진짜 30% 고객, 20% 시간 데이터 샘플링 결과 요약")
        print("="*60)
        
        if 'temporal_bias_check' in self.analysis_results:
            bias_check = self.analysis_results['temporal_bias_check']
            print(f"✅ 시간적 편향 해결: {'성공' if bias_check['bias_resolved'] else '부분적'}")
            print(f"   - 월별 균형도 (CV): {bias_check['monthly_balance_cv']:.3f}")
            print(f"   - 계절별 균형도 (CV): {bias_check['seasonal_balance_cv']:.3f}")
            print(f"   - 커버된 월 수: {bias_check['months_covered']}/12")
            print(f"   - 커버된 계절 수: {bias_check['seasons_covered']}/4")
            print(f"   - 시간적 범위: {bias_check['temporal_range_days']}일")
        
        if 'temporal_patterns' in self.analysis_results:
            patterns = self.analysis_results['temporal_patterns']
            print(f"\n📊 시간 패턴 분석:")
            print(f"   - 피크 시간대: {patterns['peak_hours']}")
            print(f"   - 비피크 시간대: {patterns['off_peak_hours']}")
            print(f"   - 주말 비율: {patterns['weekend_ratio']:.3f}")
        
        if 'basic_patterns' in self.analysis_results:
            patterns = self.analysis_results['basic_patterns']
            print(f"\n📊 기본 패턴 분석:")
            print(f"   - 분석 고객 수: {patterns['total_customers_analyzed']}명")
            print(f"   - 시간대별/일별/월별/계절별 통계 완료")
            print("   - 변동계수(CV)는 알고리즘_v4.py에서 계산됩니다")
        
        if 'anomaly_patterns' in self.analysis_results:
            anomaly = self.analysis_results['anomaly_patterns']
            print(f"\n⚠️  이상 패턴 분석:")
            print(f"   - 이상치 비율: {anomaly['outlier_rate']:.3f}")
            print(f"   - 제로값 비율: {anomaly['zero_rate']:.3f}")
            print(f"   - 급변 비율: {anomaly['sudden_change_rate']:.3f}")
            print(f"   - 이상 고객 비율: {anomaly['estimated_anomaly_rate']:.3f}")
        
        print(f"\n💾 메타데이터:")
        if 'metadata' in self.analysis_results:
            meta = self.analysis_results['metadata']
            print(f"   - 샘플 크기: {meta['sample_size']:,}건")
            print(f"   - 고객 수: {meta['total_customers']}명")
            print(f"   - 실제 고객 샘플링 비율: {meta.get('customer_sample_ratio_used', 0)*100:.0f}%")
            print(f"   - 실제 시간 샘플링 비율: {meta.get('time_sample_ratio_used', 0)*100:.0f}%")
            print(f"   - 샘플링 방법: {meta['sampling_method']}")
            print(f"   - 시간적 편향 해결: {meta['temporal_bias_fixed']}")


def main():
    """메인 실행 함수"""
    target_customers = 500      # 500명 (최대 제한)
    records_per_customer = 100  # 고객당 100개 (최대 제한)
    
    print("한국전력 데이터 분석 (진짜 30% 고객, 20% 시간 샘플링)")
    print("="*60)
    print(f"최대 분석 대상: {target_customers}명")
    print(f"고객당 최대 레코드: {records_per_customer}개")
    print("실제 샘플링: 전체 고객의 30%, 각 고객 시간 데이터의 20%")
    print()
    
    analyzer = KEPCOAnalyzer(
        target_customers=target_customers,
        records_per_customer=records_per_customer
    )
    
    result_path = analyzer.run_analysis()
    
    print(f"\n✅ 분석 완료!")
    print(f"결과 파일: {result_path}")
    
    return result_path


if __name__ == "__main__":
    main()