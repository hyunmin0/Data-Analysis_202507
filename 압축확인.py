"""
데이터안심구역 HDF5 압축 테스트 스크립트
LP 데이터 파일 4개로 압축 옵션 테스트
"""

import pandas as pd
import numpy as np
import os
import glob
import time
from datetime import datetime
import gc

class CompressionTester:
    def __init__(self, data_dir='./제13회 산업부 공모전 대상고객 LP데이터/'):
        self.data_dir = data_dir
        self.test_results = []
        
    def find_lp_files(self, limit=4):
        """LP 파일 찾기 (최대 4개)"""
        patterns = [
            'processed_LPData_*.csv',
            'LP데이터*.csv', 
            '*LP*.csv'
        ]
        
        lp_files = []
        for pattern in patterns:
            files = glob.glob(os.path.join(self.data_dir, pattern))
            lp_files.extend(files)
        
        # 중복 제거 및 정렬
        lp_files = list(set(lp_files))
        lp_files.sort()
        
        # 최대 4개 파일만 선택
        if len(lp_files) > limit:
            lp_files = lp_files[:limit]
            
        print(f"🔍 발견된 LP 파일 ({len(lp_files)}개):")
        for i, file in enumerate(lp_files, 1):
            file_size = os.path.getsize(file) / 1024**2  # MB
            print(f"   {i}. {os.path.basename(file)} ({file_size:.1f}MB)")
            
        return lp_files
    
    def test_compression_options(self, sample_data):
        """다양한 압축 옵션 테스트"""
        
        # 테스트할 압축 옵션들
        compression_options = [
            {'name': 'no_compression', 'complib': None, 'complevel': 0},
            {'name': 'zlib_level1', 'complib': 'zlib', 'complevel': 1},
            {'name': 'zlib_level6', 'complib': 'zlib', 'complevel': 6},
            {'name': 'zlib_level9', 'complib': 'zlib', 'complevel': 9},
            {'name': 'blosc_level5', 'complib': 'blosc', 'complevel': 5},
            {'name': 'blosc_level9', 'complib': 'blosc', 'complevel': 9},
            {'name': 'lzo_level1', 'complib': 'lzo', 'complevel': 1},
            {'name': 'bzip2_level9', 'complib': 'bzip2', 'complevel': 9},
        ]
        
        print(f"\n🧪 압축 옵션 테스트 (샘플 데이터: {len(sample_data):,}건)")
        print("=" * 80)
        print(f"{'옵션명':<15} | {'상태':<6} | {'크기(MB)':<10} | {'압축률':<8} | {'저장시간':<8} | {'읽기시간':<8}")
        print("-" * 80)
        
        baseline_size = None
        test_results = []
        
        for option in compression_options:
            try:
                test_file = f"./test_{option['name']}.h5"
                
                # 저장 시간 측정
                start_time = time.time()
                
                if option['complib'] is None:
                    # 압축 없음
                    sample_data.to_hdf(test_file, key='df', mode='w', format='table')
                else:
                    # 압축 적용
                    sample_data.to_hdf(test_file, key='df', mode='w', format='table',
                                     complib=option['complib'], 
                                     complevel=option['complevel'])
                
                save_time = time.time() - start_time
                
                # 파일 크기 확인
                file_size_mb = os.path.getsize(test_file) / 1024**2
                
                # 읽기 시간 측정
                start_time = time.time()
                _ = pd.read_hdf(test_file, key='df')
                read_time = time.time() - start_time
                
                # 압축률 계산
                if baseline_size is None:
                    baseline_size = file_size_mb
                    compression_ratio = "기준"
                else:
                    compression_ratio = f"{((baseline_size - file_size_mb) / baseline_size * 100):5.1f}%"
                
                print(f"{option['name']:<15} | {'✅':<6} | {file_size_mb:8.2f}  | {compression_ratio:<8} | {save_time:6.3f}s | {read_time:6.3f}s")
                
                test_results.append({
                    'name': option['name'],
                    'success': True,
                    'file_size_mb': file_size_mb,
                    'save_time': save_time,
                    'read_time': read_time,
                    'complib': option['complib'],
                    'complevel': option['complevel']
                })
                
                # 테스트 파일 삭제
                os.remove(test_file)
                
            except Exception as e:
                print(f"{option['name']:<15} | {'❌':<6} | {'N/A':<10} | {'N/A':<8} | {'N/A':<8} | {str(e)[:15]}")
                
                test_results.append({
                    'name': option['name'],
                    'success': False,
                    'error': str(e),
                    'complib': option['complib'],
                    'complevel': option['complevel']
                })
        
        return test_results
    
    def process_lp_files_with_compression(self, lp_files):
        """LP 파일들을 압축하여 처리"""
        
        print(f"\n🚀 LP 파일 압축 처리 시작 ({len(lp_files)}개 파일)")
        print("=" * 60)
        
        # 결과 저장 디렉터리 생성
        os.makedirs('./compression_test_results', exist_ok=True)
        
        # 첫 번째 파일로 압축 옵션 테스트
        if lp_files:
            print(f"📁 첫 번째 파일로 압축 옵션 테스트: {os.path.basename(lp_files[0])}")
            
            # 샘플 데이터 로드
            sample_df = pd.read_csv(lp_files[0])
            
            # 컬럼명 정리
            if 'LP수신일자' in sample_df.columns:
                sample_df = sample_df.rename(columns={'LP수신일자': 'LP 수신일자'})
            if '순방향유효전력' in sample_df.columns:
                sample_df = sample_df.rename(columns={'순방향유효전력': '순방향 유효전력'})
            
            # datetime 처리
            sample_df = self._process_datetime(sample_df)
            
            # 압축 옵션 테스트
            compression_results = self.test_compression_options(sample_df)
            
            # 최적 압축 옵션 선택
            successful_options = [r for r in compression_results if r['success']]
            if successful_options:
                # 파일 크기가 가장 작은 옵션 선택
                best_option = min(successful_options, key=lambda x: x['file_size_mb'])
                print(f"\n🏆 최적 압축 옵션: {best_option['name']} (크기: {best_option['file_size_mb']:.2f}MB)")
            else:
                print("\n❌ 사용 가능한 압축 옵션이 없습니다!")
                return False
            
            del sample_df
            gc.collect()
        
        # 전체 파일 처리
        print(f"\n📦 전체 파일 압축 처리 시작...")
        
        total_records = 0
        all_customers = set()
        
        # 최적 압축 설정
        if best_option['complib'] is None:
            compression_settings = {}
        else:
            compression_settings = {
                'complib': best_option['complib'],
                'complevel': best_option['complevel']
            }
        
        hdf5_path = './compression_test_results/compressed_lp_data.h5'
        
        # 기존 파일 삭제
        if os.path.exists(hdf5_path):
            os.remove(hdf5_path)
        
        start_total_time = time.time()
        
        for i, file_path in enumerate(lp_files):
            try:
                filename = os.path.basename(file_path)
                print(f"\n📄 [{i+1}/{len(lp_files)}] {filename} 처리 중...")
                
                # 파일 로드
                df = pd.read_csv(file_path)
                original_size_mb = os.path.getsize(file_path) / 1024**2
                
                print(f"   📊 원본 크기: {original_size_mb:.1f}MB, 레코드: {len(df):,}개")
                
                # 데이터 정제
                df = self._clean_data(df)
                
                if len(df) == 0:
                    print(f"   ❌ 유효한 데이터 없음")
                    continue
                
                # 압축 저장
                save_start = time.time()
                
                if i == 0:
                    # 첫 번째 파일
                    if compression_settings:
                        df.to_hdf(hdf5_path, key='df', mode='w', format='table', **compression_settings)
                    else:
                        df.to_hdf(hdf5_path, key='df', mode='w', format='table')
                else:
                    # 추가 파일
                    if compression_settings:
                        df.to_hdf(hdf5_path, key='df', mode='a', format='table', append=True, **compression_settings)
                    else:
                        df.to_hdf(hdf5_path, key='df', mode='a', format='table', append=True)
                
                save_time = time.time() - save_start
                
                # 현재 파일 크기 확인
                current_size_mb = os.path.getsize(hdf5_path) / 1024**2
                
                # 통계 수집
                total_records += len(df)
                all_customers.update(df['대체고객번호'].unique())
                
                print(f"   ✅ 저장 완료: {save_time:.2f}초")
                print(f"   💾 누적 크기: {current_size_mb:.1f}MB")
                print(f"   👥 누적 고객: {len(all_customers)}명")
                
                del df
                gc.collect()
                
            except Exception as e:
                print(f"   ❌ 처리 실패: {e}")
                continue
        
        total_time = time.time() - start_total_time
        
        # 최종 결과
        if os.path.exists(hdf5_path):
            final_size_mb = os.path.getsize(hdf5_path) / 1024**2
            
            print(f"\n🎯 압축 처리 완료!")
            print("=" * 50)
            print(f"⏱️  총 처리 시간: {total_time:.1f}초")
            print(f"📊 총 레코드: {total_records:,}개")
            print(f"👥 총 고객: {len(all_customers)}명")
            print(f"💾 최종 파일 크기: {final_size_mb:.1f}MB")
            print(f"🗜️ 사용된 압축: {best_option['name']}")
            
            # 원본 대비 압축률 추정
            estimated_original_mb = total_records * 0.2 / 1000  # 대략 추정
            if estimated_original_mb > 0:
                compression_ratio = (1 - final_size_mb / estimated_original_mb) * 100
                print(f"📈 예상 압축률: {compression_ratio:.1f}%")
            
            # 읽기 테스트
            print(f"\n📖 압축된 파일 읽기 테스트...")
            read_start = time.time()
            test_sample = pd.read_hdf(hdf5_path, key='df', start=0, stop=1000)
            read_time = time.time() - read_start
            print(f"   ✅ 1000건 읽기: {read_time:.3f}초")
            print(f"   📅 데이터 범위: {test_sample['datetime'].min()} ~ {test_sample['datetime'].max()}")
            
            return True
        else:
            print("\n❌ 압축 파일 생성 실패")
            return False
    
    def _process_datetime(self, df):
        """datetime 컬럼 처리"""
        try:
            if 'LP 수신일자' in df.columns:
                # 24:00 처리
                df['LP 수신일자'] = df['LP 수신일자'].str.replace(' 24:00', ' 00:00')
                df['datetime'] = pd.to_datetime(df['LP 수신일자'], errors='coerce')
            return df
        except Exception as e:
            print(f"   ⚠️ datetime 처리 오류: {e}")
            return df
    
    def _clean_data(self, df):
        """데이터 정제"""
        try:
            # 컬럼명 표준화
            if 'LP수신일자' in df.columns:
                df = df.rename(columns={'LP수신일자': 'LP 수신일자'})
            if '순방향유효전력' in df.columns:
                df = df.rename(columns={'순방향유효전력': '순방향 유효전력'})
            
            # datetime 처리
            df = self._process_datetime(df)
            
            # 필수 컬럼 확인
            required_cols = ['대체고객번호', 'LP 수신일자', '순방향 유효전력']
            if not all(col in df.columns for col in required_cols):
                return pd.DataFrame()
            
            # 결측치 및 이상값 제거
            df = df.dropna(subset=['datetime', '순방향 유효전력'])
            df = df[df['순방향 유효전력'] >= 0]
            
            return df
            
        except Exception as e:
            print(f"   ⚠️ 데이터 정제 오류: {e}")
            return pd.DataFrame()

def main():
    """메인 실행 함수"""
    print("🗜️ 데이터안심구역 HDF5 압축 테스트 도구")
    print("=" * 60)
    print(f"시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 테스터 초기화
    tester = CompressionTester()
    
    # LP 파일 찾기
    lp_files = tester.find_lp_files(limit=4)
    
    if not lp_files:
        print("❌ LP 데이터 파일을 찾을 수 없습니다!")
        print("💡 다음 경로를 확인하세요:")
        print("   - ./제13회 산업부 공모전 대상고객 LP데이터/")
        print("   - processed_LPData_*.csv 패턴의 파일들")
        return False
    
    # 압축 테스트 실행
    success = tester.process_lp_files_with_compression(lp_files)
    
    if success:
        print(f"\n✅ 압축 테스트 성공!")
        print(f"📁 결과 파일: ./compression_test_results/compressed_lp_data.h5")
        print(f"\n💡 이제 실제 전처리 코드에서 동일한 압축 설정을 사용하세요!")
    else:
        print(f"\n❌ 압축 테스트 실패")
    
    print(f"\n완료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return success

if __name__ == "__main__":
    main()