import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import glob
import os
import gc  # ← 추가
import json  # ← 추가
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class KEPCODataAnalyzer:
    def __init__(self):
        self.customer_data = None
        self.lp_data = None
        
        self.analysis_results = {}
        
    def load_customer_data(self, file_path='제13회 산업부 공모전 대상고객/제13회 산업부 공모전 대상고객.xlsx'):
        """실제 고객 기본정보 로딩 및 기본 분석"""
        print("=== 고객 기본정보 로딩 ===")
        
        try:
            # 실제 Excel 파일 읽기
            self.customer_data = pd.read_excel(file_path, header=1)
            
            print(f"총 고객 수: {len(self.customer_data):,}명")
            print(f"컬럼: {list(self.customer_data.columns)}")
            print("\n기본 정보:")
            print(self.customer_data.head())
            
            return self._analyze_customer_distribution()
            
        except Exception as e:
            print(f"고객 데이터 로딩 실패: {e}")
            return None
    
    def _analyze_customer_distribution(self):
        """고객 분포 분석"""
        print("\n=== 고객 분포 분석 ===")
        
        # 계약종별 분포
        contract_counts = self.customer_data['계약종별'].value_counts()
        print("\n📊 계약종별 분포:")
        for contract, count in contract_counts.items():
            pct = (count / len(self.customer_data)) * 100
            print(f"  {contract}: {count}명 ({pct:.1f}%)")
        
        # 사용용도별 분포
        usage_counts = self.customer_data['사용용도'].value_counts()
        print("\n🏭 사용용도별 분포:")
        for usage, count in usage_counts.items():
            pct = (count / len(self.customer_data)) * 100
            print(f"  {usage}: {count}명 ({pct:.1f}%)")
        
        # 계약전력 분포
        print("\n⚡ 계약전력 분포:")
        power_stats = self.customer_data['계약전력'].describe()
        print(power_stats)
        
        return {
            'contract_distribution': contract_counts,
            'usage_distribution': usage_counts,
            'power_stats': power_stats
        }
    
    def load_lp_data(self, data_dir):
        """LP 데이터 로딩 - 메모리 최적화"""
        lp_files = []
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.startswith('processed_LPData_') and file.endswith('.csv'):
                    lp_files.append(os.path.join(root, file))
        
        if not lp_files:
            print("❌ LP 파일을 찾을 수 없습니다.")
            return False
        
        print(f"발견된 LP 파일 수: {len(lp_files)}개")
        
        # ⭐ 핵심 변경: 파일별로 처리하고 바로 HDF5에 저장
        os.makedirs('./analysis_results', exist_ok=True)
        total_records = 0
        all_customers = set()
        
        for i, file_path in enumerate(sorted(lp_files)):
            try:
                filename = os.path.basename(file_path)
                print(f"   [{i+1}/{len(lp_files)}] {filename} 처리 중...")
                
                # 한 파일만 메모리에 로드
                df = pd.read_csv(file_path)
                
                # 컬럼명 표준화
                if 'LP수신일자' in df.columns:
                    df = df.rename(columns={'LP수신일자': 'LP 수신일자'})
                if '순방향유효전력' in df.columns:
                    df = df.rename(columns={'순방향유효전력': '순방향 유효전력'})
                
                # 필수 컬럼 확인
                required_cols = ['대체고객번호', 'LP 수신일자', '순방향 유효전력']
                if not all(col in df.columns for col in required_cols):
                    print(f"      ❌ 필수 컬럼 누락: {filename}")
                    del df
                    continue
                
                # 데이터 정제
                df = self._process_datetime_file(df)
                df = df.dropna(subset=required_cols)
                df = df[df['순방향 유효전력'] >= 0]
                
                if len(df) == 0:
                    print(f"      ❌ 유효한 데이터 없음: {filename}")
                    del df
                    continue
                
                # 파일별 HDF5 저장 (append 모드)
                hdf5_path = './analysis_results/processed_lp_data.h5'
                try:
                    if i == 0:
                        df.to_hdf(hdf5_path, key='df', mode='w', format='table')
                    else:
                        df.to_hdf(hdf5_path, key='df', mode='a', format='table', append=True)
                except Exception as hdf_error:
                    print(f"      ⚠️ HDF5 저장 실패: {hdf_error}")
                    print(f"      💡 해결방법: pip install tables")
                    del df
                    continue
                
                # 통계만 수집
                total_records += len(df)
                all_customers.update(df['대체고객번호'].unique())
                
                print(f"      ✅ 레코드: {len(df):,}개, 고객: {df['대체고객번호'].nunique()}명")
                
                # 메모리에서 즉시 삭제
                del df
                gc.collect()
                
            except Exception as e:
                print(f"      ❌ 파일 처리 실패: {e}")
                continue
        
        if total_records == 0:
            print("❌ 처리된 데이터가 없습니다.")
            return False
        
        print(f"\n✅ LP 데이터 처리 완료:")
        print(f"  총 레코드: {total_records:,}")
        print(f"  총 고객: {len(all_customers)}명")
        print(f"  저장 위치: ./analysis_results/processed_lp_data.h5")
        
        # 전체 데이터를 메모리에 로드하지 않고 통계만 저장
        self.lp_summary = {
            'total_records': total_records,
            'total_customers': len(all_customers),
            'hdf5_file': './analysis_results/processed_lp_data.h5'
        }
        
        return True

    def _process_datetime_file(self, df):
        """전체 파일의 datetime 처리"""
        try:
            # 24:00을 00:00으로 변경
            original_24_mask = df['LP 수신일자'].str.contains(' 24:00', na=False)
            df['LP 수신일자'] = df['LP 수신일자'].str.replace(' 24:00', ' 00:00')
            
            # datetime 변환
            df['datetime'] = pd.to_datetime(df['LP 수신일자'], errors='coerce')
            
            # 24:00이었던 행들은 다음날로 이동
            if original_24_mask.any():
                df.loc[original_24_mask, 'datetime'] += pd.Timedelta(days=1)
            
            return df
        except Exception as e:
            print(f"   ⚠️ datetime 처리 오류: {e}")
            df['datetime'] = pd.to_datetime(df['LP 수신일자'], errors='coerce')
            return df
    
    def detect_outliers_streamed(self, customer_limit=3, method='iqr'):
        """HDF5 기반 스트리밍 이상치 탐지 (샘플 고객 기준)"""
        print("\n[스트리밍 기반] 이상치 탐지 (샘플 고객)")
        
        # ⭐ 수정: 올바른 HDF5 파일 경로 사용
        hdf_path = './analysis_results/processed_lp_data.h5'
        if not os.path.exists(hdf_path):
            print("❌ HDF5 파일이 존재하지 않습니다.")
            return None

        try:
            # ⭐ 수정: 간단한 방식으로 HDF5 읽기
            sample_data = pd.read_hdf(hdf_path, key='df', start=0, stop=10000)  # 첫 1만개만 샘플
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

                        summary[f"{cid}-{col}"] = {
                            'outlier_count': len(outliers),
                            'outlier_pct': pct,
                            'lower_bound': lower,
                            'upper_bound': upper
                        }

            self.analysis_results['outliers_streamed'] = summary
            return summary

        except Exception as e:
            print(f"❌ 스트리밍 이상치 분석 실패: {e}")
            import traceback
            traceback.print_exc()
            return None

    def generate_quality_report_streamed(self):
        """HDF5 기반 스트리밍 품질 리포트"""
        print("\n📋 스트리밍 기반 데이터 품질 리포트")
        print("=" * 60)

        # ⭐ 수정: 올바른 HDF5 파일 경로 사용
        hdf_path = './analysis_results/processed_lp_data.h5'
        if not os.path.exists(hdf_path):
            print("❌ HDF5 파일이 없습니다.")
            return False

        try:
            # ⭐ 수정: 간단한 방식으로 데이터 읽기
            sample_data = pd.read_hdf(hdf_path, key='df', start=0, stop=50000)  # 5만개 샘플
            
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
                'file': hdf_path
            }

            print(f"✅ 샘플 레코드: {len(sample_data):,}건")
            print(f"✅ 측정 기간: {start} ~ {end}")
            print(f"✅ 평균 유효전력: {mean_power:.2f} kW")

            # JSON 저장
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


# 사용 예제 (실제 데이터안심구역에서 실행)
if __name__ == "__main__":
    print("한국전력공사 전력 사용패턴 변동계수 개발 프로젝트")
    print("데이터안심구역 전용 - 실제 데이터 분석")
    print("="*60)
    
    # 분석기 초기화
    analyzer = KEPCODataAnalyzer()
    
    # 1단계: 고객 기본정보 분석
    print("\n[1단계] 고객 기본정보 로딩 및 분석")
    customer_analysis = analyzer.load_customer_data('제13회 산업부 공모전 대상고객/제13회 산업부 공모전 대상고객.xlsx')
    
    # 2단계: LP 데이터 분석
    print("\n[2단계] LP 데이터 로딩 및 품질 분석")
    lp_analysis = analyzer.load_lp_data('./제13회 산업부 공모전 대상고객 LP데이터/')
    
    # 3단계: 이상치 탐지 (lp_analysis 성공시에만)
    print("\n[3단계] 이상치 탐지 및 데이터 정제")
    if lp_analysis:
        outliers = analyzer.detect_outliers_streamed(customer_limit=3)
    else:
        print("❌ LP 데이터 로딩 실패로 이상치 탐지 건너뜀")
    
    # 4단계: 종합 리포트 (lp_analysis 성공시에만)
    print("\n[4단계] 데이터 품질 종합 평가")
    if lp_analysis:
        analyzer.generate_quality_report_streamed()
    else:
        print("❌ LP 데이터 로딩 실패로 품질 리포트 건너뜀")
    
    print("\n🎯 1단계 데이터 품질 점검 완료!")
    print("다음: 2단계 시계열 패턴 분석 준비 완료")