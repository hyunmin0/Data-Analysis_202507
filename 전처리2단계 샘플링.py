"""
한국전력 데이터 JSON 생성
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
        
    def load_hdf5_data(self, hdf5_path='./analysis_results/processed_lp_data.h5'):
        """ 수정: 스마트 샘플링 적용"""
        with pd.HDFStore(hdf5_path, mode='r') as store:
            total_rows = store.get_storer('df').nrows
        
        if self.sample_size >= total_rows:
            # 전체 데이터가 작으면 모두 로딩
            self.df = pd.read_hdf(hdf5_path, key='df')
        else:
            # 스마트 샘플링 적용
            self.df = self._sample_from_hdf5(hdf5_path, total_rows)
        
        self._prepare_datetime_features()
    
    def _sample_from_hdf5(self, hdf5_path, total_rows):
        """새로 추가: HDF5에서 스마트 샘플링"""
        print("    스마트 샘플링 적용 중...")
        
        # 1. 먼저 고객 정보 파악
        sample_chunk = pd.read_hdf(hdf5_path, key='df', start=0, stop=min(50000, total_rows))
        
        # 2. 고객별 데이터 분포 파악
        customer_counts = sample_chunk['대체고객번호'].value_counts()
        available_customers = customer_counts.index.tolist()
        
        # 3. 목표 고객 수만큼 선택
        target_customers = min(self.target_customers, len(available_customers))
        selected_customers = np.random.choice(
            available_customers, 
            size=target_customers, 
            replace=False
        ).tolist()
        
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
    
    def analyze_volatility(self):
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
        
        # 급격한 변화 분석 (샘플링으로)
        sudden_changes = 0
        if len(self.df) > 10000:
            sample_df = self.df.sample(n=10000, random_state=42)
            sample_df = sample_df.sort_values(['대체고객번호', 'datetime'])
            diff = sample_df.groupby('대체고객번호')[target_col].diff().abs()
            sudden_changes = (diff > diff.quantile(0.95)).sum()
        
        sudden_change_rate = sudden_changes / len(customers) if len(customers) > 0 else 0
        
        # 고객별 이상 패턴 분석
        sample_customers = customers[:min(50, len(customers))]
        anomaly_customers = {
            'high_night_usage': 0,
            'excessive_zeros': 0,
            'high_volatility': 0,
            'statistical_outliers': 0
        }
        
        for customer_id in sample_customers:
            customer_data = self.df[self.df['대체고객번호'] == customer_id]
            
            # 야간 과다 사용
            if night_day_ratio > 1.2:
                anomaly_customers['high_night_usage'] += 1
            
            # 제로값 과다
            customer_zeros = (customer_data[target_col] == 0).sum()
            if customer_zeros / len(customer_data) > 0.1:
                anomaly_customers['excessive_zeros'] += 1
            
            # 높은 변동성
            customer_cv = customer_data[target_col].std() / customer_data[target_col].mean()
            if customer_cv > 0.5:
                anomaly_customers['high_volatility'] += 1
            
            # 통계적 이상치
            customer_outliers = customer_data[
                (customer_data[target_col] < lower_bound) | 
                (customer_data[target_col] > upper_bound)
            ]
            if len(customer_outliers) / len(customer_data) > 0.05:
                anomaly_customers['statistical_outliers'] += 1
        
        # 결과 계산
        total_anomaly_customers = sum(anomaly_customers.values())
        anomaly_rate = (total_anomaly_customers / len(sample_customers)) * 100
        
        # 결과 저장 (한 번만)
        self.analysis_results['anomaly_analysis'] = {
            'processed_customers': len(sample_customers),
            'total_outliers': int(len(outliers)),
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
    
    def generate_json_result(self, output_path='./analysis_results/analysis_results2.json'):
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
    
    def run_analysis(self, hdf5_path='./analysis_results/processed_lp_data.h5'):
        self.load_hdf5_data(hdf5_path)
        self.analyze_temporal_patterns()
        self.analyze_volatility()
        self.analyze_anomalies()
        output_path = self.generate_json_result()
        
        if hasattr(self, 'df'):
            del self.df
        gc.collect()
        
        return output_path


def main():
    target_customers = 500      # 500명
    records_per_customer = 100  # 고객당 100개 (약 1일치)
    
    print(f"분석 대상: {target_customers}명")
    
    analyzer = KEPCOAnalyzer(
        target_customers=target_customers,
        records_per_customer=records_per_customer
    )
    result_path = analyzer.run_analysis()
    return result_path


if __name__ == "__main__":
    main()