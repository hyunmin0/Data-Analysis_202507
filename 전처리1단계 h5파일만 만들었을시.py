import pandas as pd
import numpy as np
import json
import os
from datetime import datetime

class QuickAnalyzer:
    def __init__(self):
        self.analysis_results = {}
        
    def detect_outliers_streamed(self, customer_limit=3, method='iqr'):
        """HDF5 기반 스트리밍 이상치 탐지 (수정된 버전)"""
        print("\n[스트리밍 기반] 이상치 탐지 (샘플 고객)")
        
        hdf_path = './analysis_results/processed_lp_data.h5'
        if not os.path.exists(hdf_path):
            print("❌ HDF5 파일이 존재하지 않습니다.")
            return None

        try:
            # 샘플 데이터 읽기
            sample_data = pd.read_hdf(hdf_path, key='df', start=0, stop=10000)
            customer_ids = sample_data['대체고객번호'].unique()
            print(f"💡 샘플 고객 수: {len(customer_ids)}명 중 {customer_limit}명 분석")

            summary = {}

            for cid in customer_ids[:customer_limit]:
                df = sample_data[sample_data['대체고객번호'] == cid]
                print(f"\n📌 고객 {cid} - 레코드 수: {len(df):,}")

                numeric_columns = ['순방향 유효전력', '지상무효', '진상무효', '피상전력']
                available_cols = [col for col in numeric_columns if col in df.columns]

                for col in available_cols:
                    if method == 'iqr':
                        Q1 = df[col].quantile(0.25)
                        Q3 = df[col].quantile(0.75)
                        IQR = Q3 - Q1
                        lower = Q1 - 1.5 * IQR
                        upper = Q3 + 1.5 * IQR
                        outliers = df[(df[col] < lower) | (df[col] > upper)]

                        pct = len(outliers) / len(df) * 100 if len(df) > 0 else 0
                        print(f"   - {col}: 이상치 {len(outliers)}건 ({pct:.2f}%)")

                        # ⭐ 수정: tuple 대신 문자열 키 사용
                        summary[f"{cid}_{col}"] = {
                            'customer': cid,
                            'column': col,
                            'outlier_count': len(outliers),
                            'outlier_pct': pct,
                            'lower_bound': float(lower),
                            'upper_bound': float(upper)
                        }

            self.analysis_results['outliers_streamed'] = summary
            return summary

        except Exception as e:
            print(f"❌ 스트리밍 이상치 분석 실패: {e}")
            import traceback
            traceback.print_exc()
            return None

    def generate_quality_report_streamed(self):
        """HDF5 기반 스트리밍 품질 리포트 (수정된 버전)"""
        print("\n📋 스트리밍 기반 데이터 품질 리포트")
        print("=" * 60)

        hdf_path = './analysis_results/processed_lp_data.h5'
        if not os.path.exists(hdf_path):
            print("❌ HDF5 파일이 없습니다.")
            return False

        try:
            # 샘플 데이터 읽기
            sample_data = pd.read_hdf(hdf_path, key='df', start=0, stop=50000)
            
            # 날짜 범위
            start = sample_data['datetime'].min()
            end = sample_data['datetime'].max()
            
            # 평균 유효전력 계산
            mean_power = sample_data['순방향 유효전력'].mean()
            
            # 요약 저장
            self.analysis_results['lp_summary_streamed'] = {
                'date_range': {'start': str(start), 'end': str(end)},
                'avg_power': float(mean_power),
                'sample_records': len(sample_data),
                'total_customers': int(sample_data['대체고객번호'].nunique()),
                'file': hdf_path
            }

            print(f"✅ 샘플 레코드: {len(sample_data):,}건")
            print(f"✅ 샘플 고객 수: {sample_data['대체고객번호'].nunique()}명")
            print(f"✅ 측정 기간: {start} ~ {end}")
            print(f"✅ 평균 유효전력: {mean_power:.2f} kW")

            # 전체 파일 크기 정보
            file_size = os.path.getsize(hdf_path) / 1024**3
            print(f"✅ HDF5 파일 크기: {file_size:.2f} GB")

            # JSON 저장 (수정된 버전)
            output_file = './analysis_results/analysis_results.json'
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(self.analysis_results, f, indent=2, ensure_ascii=False, default=str)

            print(f"💾 결과 저장 완료: {output_file}")
            return True

        except Exception as e:
            print(f"❌ 품질 리포트 생성 실패: {e}")
            import traceback
            traceback.print_exc()
            return False

    def get_full_data_info(self):
        """전체 데이터 정보 확인 (샘플링 없이)"""
        print("\n📊 전체 HDF5 파일 정보")
        print("=" * 40)
        
        hdf_path = './analysis_results/processed_lp_data.h5'
        
        try:
            # HDF5 파일 정보만 확인
            with pd.HDFStore(hdf_path, mode='r') as store:
                info = store.info()
                print("📁 HDF5 파일 구조:")
                print(info)
                
                # 전체 크기 확인
                nrows = store.get_storer('df').nrows
                print(f"\n📈 전체 데이터:")
                print(f"  총 레코드: {nrows:,}건")
                
                # 메모리 안전한 방식으로 고객 수 확인
                unique_customers = store.select_column('df', '대체고객번호').nunique()
                print(f"  총 고객 수: {unique_customers:,}명")
                
        except Exception as e:
            print(f"❌ 파일 정보 확인 실패: {e}")

# ============ 실행 코드 ============
if __name__ == "__main__":
    print("🔧 HDF5 기반 빠른 분석 재실행")
    print("=" * 50)
    
    analyzer = QuickAnalyzer()
    
    # 1. 전체 데이터 정보 확인
    analyzer.get_full_data_info()
    
    # 2. 이상치 탐지 재실행 (수정된 버전)
    print("\n[1단계] 이상치 탐지 재실행")
    outliers = analyzer.detect_outliers_streamed(customer_limit=3)
    
    # 3. 품질 리포트 재실행 (수정된 버전)
    print("\n[2단계] 품질 리포트 재실행")
    result = analyzer.generate_quality_report_streamed()
    
    if result:
        print("\n✅ 에러 수정 완료!")
        print("📁 생성된 파일:")
        print("  - ./analysis_results/analysis_results.json")
        print("  - ./analysis_results/processed_lp_data.h5")
    else:
        print("\n❌ 여전히 에러가 있습니다.")
    
    print("\n🎯 빠른 수정 완료!")