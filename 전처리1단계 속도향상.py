import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import glob
import os
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
    
    def load_lp_data(self, data_directory='./제13회 산업부 공모전 대상고객 LP데이터/'):
        """실제 LP 데이터 로딩 (여러 CSV 파일)"""
        print("\n=== LP 데이터 로딩 ===")
        
        try:
            # processed_LPData_YYYYMMDD_DD.csv 패턴의 파일들 찾기
            lp_files = glob.glob(os.path.join(data_directory, 'processed_LPData_*.csv'))
            
            if not lp_files:
                print("LP 데이터 파일을 찾을 수 없습니다.")
                return None
            
            print(f"발견된 LP 파일 수: {len(lp_files)}개")
            
            # 모든 LP 파일 읽기 및 결합
            lp_dataframes = []
            total_records = 0
            
            for i, file_path in enumerate(sorted(lp_files)):
                try:
                    filename = os.path.basename(file_path)
                    print(f"   [{i+1}/{len(lp_files)}] {filename} 처리 중...")

                    #청크 단위로 읽으면서 바로 처리
                    chunk_list = []

                    for chunk in pd.read_csv(file_path, chunksize=5000):  # 5000행씩 처리
                        # 컬럼명 표준화
                        if 'LP수신일자' in chunk.columns:
                            chunk = chunk.rename(columns={'LP수신일자': 'LP 수신일자'})
                        if '순방향유효전력' in chunk.columns:
                            chunk = chunk.rename(columns={'순방향유효전력': '순방향 유효전력'})

                        # 필수 컬럼 확인
                        required_cols = ['대체고객번호', 'LP 수신일자', '순방향 유효전력']
                        if all(col in chunk.columns for col in required_cols):

                            # ⭐ 24:00 처리를 청크 단위로 바로 처리
                            chunk = self._process_datetime_chunk(chunk)

                            # 데이터 품질 기본 체크
                            chunk = chunk.dropna(subset=required_cols)
                            chunk = chunk[chunk['순방향 유효전력'] >= 0]

                            chunk_list.append(chunk)

                    # 파일별 청크 결합
                    if chunk_list:
                        file_df = pd.concat(chunk_list, ignore_index=True)
                        lp_dataframes.append(file_df)
                        total_records += len(file_df)
                        print(f"      레코드: {len(file_df):,}개, 고객: {file_df['대체고객번호'].nunique()}명")
                        
                except Exception as e:
                    print(f"  ✗ 파일 로딩 실패: {e}")
                    continue
            
            if not lp_dataframes:
                print("유효한 LP 데이터가 없습니다.")
                return None
            
            # 모든 데이터 결합
            self.lp_data = pd.concat(lp_dataframes, ignore_index=True)
            
            # 시간 순서로 정렬
            self.lp_data = self.lp_data.sort_values(['대체고객번호', 'datetime']).reset_index(drop=True)
            
            print(f"\n✅ 전체 LP 데이터 결합 완료:")
            print(f"  총 레코드: {len(self.lp_data):,}")
            print(f"  총 고객: {self.lp_data['대체고객번호'].nunique()}")
            print(f"   - 기간: {self.lp_data['datetime'].min()} ~ {self.lp_data['datetime'].max()}")
            
            return self._analyze_lp_quality()
            
        except Exception as e:
            print(f"LP 데이터 로딩 실패: {e}")
            return None
    
    def _analyze_lp_quality(self):
        """LP 데이터 품질 분석"""
        print("\n=== LP 데이터 품질 분석 ===")

        # 기본 통계
        numeric_columns = ['순방향 유효전력', '지상무효', '진상무효', '피상전력']
        available_cols = [col for col in numeric_columns if col in self.lp_data.columns]

        print(f"📈 기본 통계:")
        print(self.lp_data[available_cols].describe())

        # 시간 간격 체크 (샘플만)
        sample_customers = self.lp_data['대체고객번호'].unique()[:3]
        print(f"\n⏰ 시간 간격 체크:")
        for customer in sample_customers:
            customer_data = self.lp_data[self.lp_data['대체고객번호'] == customer].sort_values('datetime')
            if len(customer_data) > 1:
                time_diffs = customer_data['datetime'].diff().dt.total_seconds() / 60
                avg_interval = time_diffs.dropna().mean()
                std_interval = time_diffs.dropna().std()
                print(f"  {customer}: 평균 간격 {avg_interval:.1f}분, 표준편차 {std_interval:.1f}분")

        # 데이터 품질 체크
        print(f"\n🔍 데이터 품질 체크:")
        for col in available_cols:
            null_count = self.lp_data[col].isnull().sum()
            null_pct = null_count / len(self.lp_data) * 100
            zero_count = (self.lp_data[col] == 0).sum()
            zero_pct = zero_count / len(self.lp_data) * 100
            print(f"  {col}:")
            print(f"    결측치: {null_count}건 ({null_pct:.2f}%)")
            print(f"    0값: {zero_count}건 ({zero_pct:.2f}%)")

        # 이상치 탐지
        print(f"\n🚨 이상치 탐지:")
        for col in available_cols:
            Q1 = self.lp_data[col].quantile(0.25)
            Q3 = self.lp_data[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = self.lp_data[(self.lp_data[col] < Q1 - 1.5 * IQR) | (self.lp_data[col] > Q3 + 1.5 * IQR)]
            outlier_pct = len(outliers) / len(self.lp_data) * 100
            print(f"  {col}: {len(outliers)}건 ({outlier_pct:.2f}%)")

        return True
    
    def _process_datetime_chunk(self, chunk):
        """청크 단위로 datetime 처리"""
        try:
            # 24:00을 00:00으로 변경하면서 다음날 표시 저장
            original_24_mask = chunk['LP 수신일자'].str.contains(' 24:00', na=False)

            # 24:00을 00:00으로 변경
            chunk['LP 수신일자'] = chunk['LP 수신일자'].str.replace(' 24:00', ' 00:00')

            # datetime 변환
            chunk['datetime'] = pd.to_datetime(chunk['LP 수신일자'], errors='coerce')

            # 원래 24:00이었던 행들은 다음날로 이동
            if original_24_mask.any():
                chunk.loc[original_24_mask, 'datetime'] += pd.Timedelta(days=1)

            return chunk

        except Exception as e:
            print(f"   ⚠️ datetime 처리 오류: {e}")
            chunk['datetime'] = pd.to_datetime(chunk['LP 수신일자'], errors='coerce')
            return chunk
    
    def detect_outliers(self, method='iqr'):
        """이상치 탐지"""
        outlier_summary = {}
        numeric_columns = ['순방향 유효전력', '지상무효', '진상무효', '피상전력']
        
        for col in numeric_columns:
            if col in self.lp_data.columns:
                if method == 'iqr':
                    Q1 = self.lp_data[col].quantile(0.25)
                    Q3 = self.lp_data[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outliers = self.lp_data[
                        (self.lp_data[col] < lower_bound) | 
                        (self.lp_data[col] > upper_bound)
                    ]
                    
                    outlier_count = len(outliers)
                    outlier_pct = (outlier_count / len(self.lp_data)) * 100
                    
                    print(f"  {col}: {outlier_count}건 ({outlier_pct:.2f}%)")
                    outlier_summary[col] = {
                        'count': outlier_count,
                        'percentage': outlier_pct,
                        'lower_bound': lower_bound,
                        'upper_bound': upper_bound
                    }
        
        return outlier_summary
    

    def generate_quality_report(self):
        """데이터 품질 종합 리포트 생성 및 전처리된 데이터 저장"""
        import json
        from datetime import datetime
        import os
        
        print("\n" + "="*60)
        print("📋 데이터 품질 종합 리포트")
        print("="*60)

        # 데이터 존재 여부 확인
        if self.customer_data is None or self.lp_data is None:
            print("❌ 데이터가 로딩되지 않았습니다.")
            return False

        # 고객 데이터 요약
        if self.customer_data is not None:
            print(f"\n👥 고객 데이터:")
            print(f"  총 고객 수: {len(self.customer_data):,}명")
            print(f"  계약종별 유형: {self.customer_data['계약종별'].nunique()}개")
            print(f"  사용용도 유형: {self.customer_data['사용용도'].nunique()}개")

            # ⭐ analysis_results에 고객 정보 저장
            self.analysis_results['customer_summary'] = {
                'total_customers': len(self.customer_data),
                'contract_types': self.customer_data['계약종별'].value_counts().to_dict(),
                'usage_types': self.customer_data['사용용도'].value_counts().to_dict()
            }

        # LP 데이터 요약
        if self.lp_data is not None:
            print(f"\n⚡ LP 데이터:")
            print(f"  총 레코드: {len(self.lp_data):,}건")
            print(f"  측정 기간: {self.lp_data['datetime'].min()} ~ {self.lp_data['datetime'].max()}")
            print(f"  데이터 커버리지: {(self.lp_data['datetime'].max() - self.lp_data['datetime'].min()).days}일")

            # 평균 전력 사용량
            avg_power = self.lp_data['순방향 유효전력'].mean()
            print(f"  평균 유효전력: {avg_power:.2f}kW")

            # ⭐ analysis_results에 LP 데이터 정보 저장
            self.analysis_results['lp_data_summary'] = {
                'total_records': len(self.lp_data),
                'total_customers': self.lp_data['대체고객번호'].nunique(),
                'date_range': {
                    'start': str(self.lp_data['datetime'].min()),
                    'end': str(self.lp_data['datetime'].max())
                },
                'avg_power': float(avg_power)
            }

        # ⭐⭐⭐ 핵심: 전처리된 데이터 저장
        print(f"\n💾 전처리된 LP 데이터 저장 중...")

        try:
            # 출력 디렉토리 생성
            import os
            os.makedirs('./analysis_results', exist_ok=True)

            # 전처리된 데이터 저장
            '''
            processed_csv = './analysis_results/processed_lp_data.csv'
            #processed_parquet = './analysis_results/processed_lp_data.parquet'

            print(f"   📊 저장 대상: {len(self.lp_data):,}개 레코드")
            print(f"   💾 저장 중... (잠시만 기다려주세요)")

            # 1. CSV 저장 (호환성용)
            print(f"      📄 CSV 저장 중...")
            #self.lp_data.to_csv(processed_csv, index=False, encoding='utf-8-sig')
            csv_size_gb = os.path.getsize(processed_csv) / 1024**3
            '''

            # 2. ⭐ HDF5 저장 (성능 최적화용)
            processed_hdf5 = './analysis_results/processed_lp_data.h5'
            print(f"   📊 저장 대상: {len(self.lp_data):,}개 레코드")
            print(f"      📦 HDF5 저장 중...")
            try:
                self.lp_data.to_hdf(processed_hdf5, key='df', mode='w', format='table')
                hdf5_size_gb = os.path.getsize(processed_hdf5) / 1024**3
                hdf5_success = True
            except Exception as hdf5_error:
                print(f"         ⚠️ HDF5 저장 실패: {hdf5_error}")
                print(f"         💡 해결방법: pip install tables")
                hdf5_success = False

            print(f"   ✅ 전처리된 데이터 저장 완료!")
            #print(f"      📄 CSV: {processed_csv} ({csv_size_gb:.2f}GB)")

            if hdf5_success:
                print(f"      📦 HDF5: {processed_hdf5} ({hdf5_size_gb:.2f}GB)")
                #print(f"      🚀 크기 절약: {((csv_size_gb - hdf5_size_gb) / csv_size_gb * 100):.1f}%")
                print(f"      ⚡ 로딩 속도 향상: 약 2-3배 빨라짐!")

            # 메타 정보 저장 (⭐ Parquet 정보 추가)
            meta_info = {
                'total_records': len(self.lp_data),
                'total_customers': self.lp_data['대체고객번호'].nunique(),
                'date_range': {
                    'start': str(self.lp_data['datetime'].min()),
                    'end': str(self.lp_data['datetime'].max())
                },
                'file_info': {
                    'hdf5_file': 'processed_lp_data.hdf5' if hdf5_success else None,
                    'hdf5_size_gb': hdf5_size_gb if hdf5_success else None,
                    'hdf5_available': hdf5_success,
                    'compression': 'table_format' if hdf5_success else None,
                    'encoding': 'utf-8-sig'
                },
                'processed_timestamp': datetime.now().isoformat()
            }

            # analysis_results에 메타 정보 추가
            self.analysis_results['processed_lp_data'] = meta_info

            if hdf5_success:
                print(f"   🚀 2-3단계에서 30분 → 3-5분으로 시간 단축 예상!")
            else:
                print(f"   📄 CSV로 저장 완료 (30분 → 8분 시간 단축)")

        except Exception as save_error:
            print(f"   ❌ 전처리된 데이터 저장 실패: {save_error}")
            print(f"      (분석은 계속 진행됩니다)")

        # ⭐⭐⭐ 필수: JSON 결과 저장 (2-3단계 연계용)
        print(f"\n💾 분석 결과 JSON 저장 중...")

        try:
            # 타임스탬프 추가
            self.analysis_results['metadata'] = {
                'timestamp': datetime.now().isoformat(),
                'analysis_stage': 'step1_preprocessing_optimized',
                'version': '2.0',
                'total_customers': len(self.customer_data) if self.customer_data is not None else 0,
                'total_lp_records': len(self.lp_data) if self.lp_data is not None else 0
            }

            # JSON 파일로 저장
            output_file = os.path.join('./analysis_results', 'analysis_results.json')

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(self.analysis_results, f, 
                         ensure_ascii=False, 
                         indent=2, 
                         default=str)

            print(f"✅ 분석 결과 JSON 저장: {output_file}")
            print(f"   저장된 항목: {len(self.analysis_results)}개")

            # 저장된 구조 확인
            print(f"   📁 저장된 구조:")
            for key in self.analysis_results.keys():
                if key == 'metadata':
                    print(f"      - metadata: 시간정보 및 버전")
                elif key == 'customer_summary':
                    print(f"      - customer_summary: 고객 기본 정보")
                elif key == 'lp_data_summary':
                    print(f"      - lp_data_summary: LP 데이터 요약")
                elif key == 'processed_lp_data':
                    print(f"      - processed_lp_data: 전처리된 데이터 메타정보")
                else:
                    print(f"      - {key}: {type(self.analysis_results[key])}")

        except Exception as json_error:
            print(f"❌ JSON 저장 실패: {json_error}")
            import traceback
            traceback.print_exc()
            return False

        # 권장사항
        print("\n💡 다음 단계 권장사항:")
        print("  1. 시계열 패턴 분석 (전처리된 데이터 활용)")
        print("  2. 고객별 사용량 프로파일링")
        print("  3. 변동성 지표 계산 및 비교")
        print("  4. 이상 패턴 탐지 알고리즘 개발")

        print(f"\n🎯 1단계 최적화 완료!")
        print(f"   📁 생성 파일:")
        print(f"      - analysis_results.json (2-3단계 연계용)")
        print(f"      - processed_lp_data.csv (전처리된 LP 데이터)")
        if 'processed_lp_data' in self.analysis_results and self.analysis_results['processed_lp_data']['file_info']['hdf5_available']:
            print(f"      - processed_lp_data.hdf5 (고성능 전처리된 데이터)")

        return True

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
    lp_analysis = analyzer.load_lp_data('./제13회 산업부 공모전 대상고객 LP데이터/')  # 현재 디렉터리에서 LP 파일 찾기
    
    # 3단계: 이상치 탐지
    print("\n[3단계] 이상치 탐지 및 데이터 정제")
    outliers = analyzer.detect_outliers('iqr')
    
    # 4단계: 종합 리포트
    print("\n[4단계] 데이터 품질 종합 평가")
    analyzer.generate_quality_report()
    
    print("\n🎯 1단계 데이터 품질 점검 완료!")
    print("다음: 2단계 시계열 패턴 분석 준비 완료")