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

class FastKEPCOJSONGenerator:
    
    def __init__(self, target_customers=500, records_per_customer=100, n_jobs=-1):
        self.target_customers = target_customers      # 500명
        self.records_per_customer = records_per_customer  # 고객당 100개
        self.sample_size = target_customers * records_per_customer  # 총 50,000개
        
        self.n_jobs = cpu_count() if n_jobs == -1 else n_jobs
        self.analysis_results = {}
        
    def load_hdf5_data(self, hdf5_path='./analysis_results/processed_lp_data.h5'):
        """스마트 샘플링 적용"""
        with pd.HDFStore(hdf5_path, mode='r') as store:
            total_rows = store.get_storer('df').nrows
        
        print(f"   전체 데이터: {total_rows:,}건")
        print(f"   목표: {self.target_customers}명 × {self.records_per_customer}개 = {self.sample_size:,}건")
        
        if self.sample_size >= total_rows:
            # 전체 데이터가 작으면 모두 로딩
            self.df = pd.read_hdf(hdf5_path, key='df')
        else:
            # 스마트 샘플링 적용
            self.df = self._smart_sampling_from_hdf5(hdf5_path, total_rows)
        
        self._prepare_datetime_features()
        
        print(f"   최종 로딩: {len(self.df):,}건")
        print(f"   고객 수: {self.df['대체고객번호'].nunique()}명")
    
    def _smart_sampling_from_hdf5(self, hdf5_path, total_rows):
        """HDF5에서 스마트 샘플링"""
        print("   스마트 샘플링 적용 중...")
        
        # 1. 먼저 고객 정보 파악 (일부 데이터만 읽어서)
        sample_chunk = pd.read_hdf(hdf5_path, key='df', start=0, stop=min(50000, total_rows))
        
        # 2. 고객별 데이터 분포 파악
        customer_counts = sample_chunk['대체고객번호'].value_counts()
        available_customers = customer_counts.index.tolist()
        
        print(f"      발견된 고객: {len(available_customers)}명")
        
        # 3. 목표 고객 수만큼 선택
        target_customers = min(self.target_customers, len(available_customers))
        selected_customers = np.random.choice(
            available_customers, 
            size=target_customers, 
            replace=False
        ).tolist()
        
        print(f"      선택된 고객: {len(selected_customers)}명")
        
        # 4. 선택된 고객들의 데이터만 로딩
        chunks = []
        chunk_size = 10000  # 한 번에 읽을 크기
        
        for start in range(0, total_rows, chunk_size):
            end = min(start + chunk_size, total_rows)
            chunk = pd.read_hdf(hdf5_path, key='df', start=start, stop=end)
            
            # 선택된 고객만 필터링
            filtered_chunk = chunk[chunk['대체고객번호'].isin(selected_customers)]
            
            if len(filtered_chunk) > 0:
                chunks.append(filtered_chunk)
            
            # 메모리 관리
            del chunk
            
            # 목표량 달성하면 중단
            if sum(len(c) for c in chunks) >= self.sample_size:
                print(f"      목표량 달성, 조기 종료")
                break
        
        # 5. 결합 및 최종 샘플링
        combined_df = pd.concat(chunks, ignore_index=True)
        
        # 6. 각 고객별로 균등하게 샘플링
        final_chunks = []
        for customer_id in selected_customers:
            customer_data = combined_df[combined_df['대체고객번호'] == customer_id]
            
            if len(customer_data) <= self.records_per_customer:
                # 데이터가 적으면 모두 사용
                final_chunks.append(customer_data)
            else:
                # 균등 간격으로 샘플링
                indices = np.linspace(0, len(customer_data)-1, self.records_per_customer, dtype=int)
                sampled_data = customer_data.iloc[indices]
                final_chunks.append(sampled_data)
        
        return pd.concat(final_chunks, ignore_index=True)
    
    def _prepare_datetime_features(self):
        """datetime 관련 특성 생성"""
        if 'datetime' not in self.df.columns:
            self.df['datetime'] = pd.to_datetime(self.df['LP 수신일자'], errors='coerce')
        
        self.df['hour'] = self.df['datetime'].dt.hour
        self.df['day_of_week'] = self.df['datetime'].dt.dayofweek
        self.df['month'] = self.df['datetime'].dt.month
        self.df['season'] = self.df['month'].map({12: '겨울', 1: '겨울', 2: '겨울',
                                                 3: '봄', 4: '봄', 5: '봄',
                                                 6: '여름', 7: '여름', 8: '여름',
                                                 9: '가을', 10: '가을', 11: '가을'})
        self.df['is_weekend'] = self.df['day_of_week'].isin([5, 6]).astype(int)
    
    def analyze_temporal_patterns(self):
        """시간대별 패턴 분석"""
        print("   시간대별 패턴 분석 중...")
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
        
        print("   시간대별 패턴 분석 완료")
    
    def analyze_volatility_parallel(self):
        """변동성 분석 (병렬 처리)"""
        print("   변동성 분석 중...")
        target_col = '순방향 유효전력'
        
        overall_cv = self.df[target_col].std() / self.df[target_col].mean()
        
        hourly_volatility = self.df.groupby('hour')[target_col].agg(['mean', 'std', 'count'])
        hourly_volatility['cv'] = hourly_volatility['std'] / hourly_volatility['mean']
        
        daily_volatility = self.df.groupby('day_of_week')[target_col].agg(['mean', 'std', 'count'])
        daily_volatility['cv'] = daily_volatility['std'] / daily_volatility['mean']
        
        monthly_volatility = self.df.groupby('month')[target_col].agg(['mean', 'std', 'count'])
        monthly_volatility['cv'] = monthly_volatility['std'] / monthly_volatility['mean']
        
        customers = self.df['대체고객번호'].unique()
        customer_cvs = self._calculate_customer_cv_chunk(target_col, customers)
            
        cv_values = list(customer_cvs.values())
        customer_cv_stats = {
            'count': len(cv_values),
            'mean': float(np.mean(cv_values)),
            'std': float(np.std(cv_values)),
            'percentiles': {
                '10%': float(np.percentile(cv_values, 10)),
                '25%': float(np.percentile(cv_values, 25)),
                '50%': float(np.percentile(cv_values, 50)),
                '75%': float(np.percentile(cv_values, 75)),
                '90%': float(np.percentile(cv_values, 90))
            }
        }
        
        cv_array = np.array(cv_values)
        volatility_distribution = {
            '매우 안정 (<0.1)': int(np.sum(cv_array < 0.1)),
            '안정 (0.1-0.2)': int(np.sum((cv_array >= 0.1) & (cv_array < 0.2))),
            '보통 (0.2-0.3)': int(np.sum((cv_array >= 0.2) & (cv_array < 0.3))),
            '높음 (0.3-0.5)': int(np.sum((cv_array >= 0.3) & (cv_array < 0.5))),
            '매우 높음 (0.5-1.0)': int(np.sum((cv_array >= 0.5) & (cv_array < 1.0))),
            '극히 높음 (>1.0)': int(np.sum(cv_array >= 1.0))
        }
        
        self.analysis_results['volatility_analysis'] = {
            'overall_cv': float(overall_cv),
            'hourly_volatility': hourly_volatility.round(6).to_dict(),
            'daily_volatility': daily_volatility.round(6).to_dict(),
            'monthly_volatility': monthly_volatility.round(6).to_dict(),
            'customer_cv_stats': customer_cv_stats,
            'volatility_distribution': volatility_distribution
        }
        
        print("   변동성 분석 완료")
    
    def _calculate_customer_cv_chunk(self, target_col, customer_chunk):
        """고객 청크별 변동계수 계산"""
        results = {}
        for customer_id in customer_chunk:
            customer_data = self.df[self.df['대체고객번호'] == customer_id][target_col]
            if len(customer_data) > 1:
                cv = customer_data.std() / customer_data.mean()
                if not np.isnan(cv) and np.isfinite(cv):
                    results[customer_id] = float(cv)
        return results
    
    def analyze_anomalies_fast(self):
        """이상 패턴 고속 분석"""
        print("   이상 패턴 분석 중...")
        target_col = '순방향 유효전력'
        customers = self.df['대체고객번호'].unique()
        
        Q1 = self.df[target_col].quantile(0.25)
        Q3 = self.df[target_col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = self.df[(self.df[target_col] < lower_bound) | (self.df[target_col] > upper_bound)]
        
        night_hours = [22, 23, 0, 1, 2, 3, 4, 5]
        day_hours = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
        
        night_data = self.df[self.df['hour'].isin(night_hours)]
        day_data = self.df[self.df['hour'].isin(day_hours)]
        
        night_mean = night_data[target_col].mean()
        day_mean = day_data[target_col].mean()
        night_day_ratio = night_mean / day_mean if day_mean > 0 else 0
        
        zero_count = (self.df[target_col] == 0).sum()
        zero_rate = zero_count / len(self.df)
        
        sudden_changes = 0
        if len(self.df) > 10000:
            sample_df = self.df.sample(n=10000, random_state=42)
            sample_df = sample_df.sort_values(['대체고객번호', 'datetime'])
            diff = sample_df[target_col].diff().abs()
            sudden_threshold = diff.quantile(0.95)
            sudden_changes = (diff > sudden_threshold).sum()
        
        sudden_change_rate = sudden_changes / len(self.df) if len(self.df) > 0 else 0
        
        # 이상 고객 감지 (변동계수 기준)
        customer_cvs = self._calculate_customer_cv_chunk(target_col, customers)
        cv_values = list(customer_cvs.values())
        cv_threshold = np.percentile(cv_values, 95) if cv_values else 1.0
        anomaly_customers = [cid for cid, cv in customer_cvs.items() if cv > cv_threshold]
        anomaly_rate = len(anomaly_customers) / len(customers) if len(customers) > 0 else 0
        
        self.analysis_results['anomaly_analysis'] = {
            'outlier_count': int(len(outliers)),
            'outlier_rate': float(len(outliers) / len(self.df)),
            'zero_count': int(zero_count),
            'zero_rate': float(zero_rate),
            'sudden_change_count': int(sudden_changes),
            'sudden_change_rate': float(sudden_change_rate),
            'night_day_ratio': float(night_day_ratio),
            'anomaly_customers': anomaly_customers,
            'estimated_anomaly_rate': float(anomaly_rate)
        }
        
        print("   이상 패턴 분석 완료")
        return True
    
    def generate_json_result(self, output_path='./analysis_results/analysis_results2.json'):
        """결과를 JSON으로 저장"""
        self.analysis_results['metadata'] = {
            'timestamp': datetime.now().isoformat(),
            'stage': 'step2_smart_sampling_analysis',
            'version': '3.0_smart_sampling',
            'sample_size': len(self.df) if hasattr(self, 'df') else 0,
            'total_customers': self.df['대체고객번호'].nunique() if hasattr(self, 'df') else 0,
            'target_customers': self.target_customers,
            'records_per_customer': self.records_per_customer,
            'sampling_method': 'smart_customer_based',
            'processing_cores': self.n_jobs
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.analysis_results, f, ensure_ascii=False, indent=2, default=str)
        
        return output_path
    
    def run_fast_analysis(self, hdf5_path='./analysis_results/processed_lp_data.h5'):
        """전체 분석 실행"""
        print("2단계 고속 분석 시작...")
        
        self.load_hdf5_data(hdf5_path)
        self.analyze_temporal_patterns()
        self.analyze_volatility_parallel()
        self.analyze_anomalies_fast()
        output_path = self.generate_json_result()
        
        if hasattr(self, 'df'):
            del self.df
        gc.collect()
        
        print(f"분석 완료. 결과 저장: {output_path}")
        return output_path


def main():
    """메인 실행 함수"""
    target_customers = 500      # 500명
    records_per_customer = 100  # 고객당 100개 (약 1일치)
    
    print("한국전력 데이터 고속 JSON 생성기")
    print("="*50)
    print(f"목표: {target_customers}명 × {records_per_customer}개 = {target_customers * records_per_customer:,}건")
    
    analyzer = FastKEPCOJSONGenerator(
        target_customers=target_customers,
        records_per_customer=records_per_customer
    )
    result_path = analyzer.run_fast_analysis()
    
    print(f"\n2단계 전처리 완료")
    print(f"결과 파일: {result_path}")
    return result_path


if __name__ == "__main__":
    main()