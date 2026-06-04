import serial  # 시리얼 통신 라이브러리 호출
import time    

# 레지스터 주소 설명: RH56F1 인간형 5지 덱스터러스 핸드 사용자 설명서 11쪽, 2.4 레지스터 설명 참조
regdict = {
    'ID'         : 1000,  # ID
    'baudrate'   : 1001,  # 보드레이트 설정
    'mode'       : 1100,  # 손가락 제어 모드(속도/힘 보호, 힘 폐루프, 임피던스, 0-1-2)
    'clearErr'   : 1003,  # 오류 지우기
    'forceClb'   : 1007,  # 힘 센서 보정
    'angleSet'   : 1040,  # 각 자유도의 각도 설정값
    'forceSet'   : 1046,  # 각 자유도의 힘 제어 임계값 설정
    'speedSet'   : 1052,  # 각 자유도의 속도 설정값
    'angleAct'   : 1064,  # 각 자유도의 실제 각도값
    'forceAct'   : 1070,  # 각 손가락에 실제로 작용하는 힘
    'errCode'    : 1082,  # 각 자유도의 전동 실린더 오류 정보
    'statusCode' : 1088,  # 각 자유도의 상태 정보
    'temp'       : 1094,  # 각 자유도 전동 실린더의 온도
    'ip'         : 1700,  #ip
    'actionSeq'  : 2160,  # 현재 동작 시퀀스 인덱스 번호
    'actionRun'  : 2162   # 현재 동작 시퀀스 실행
}

# 함수 설명: 시리얼 포트 번호와 보드레이트를 설정하고 포트를 연다. 매개변수: port는 시리얼 포트 번호, baudrate는 보드레이트
def openSerial(port, baudrate):
    ser = serial.Serial() # 시리얼 통신 함수 호출
    ser.port = port
    ser.baudrate = baudrate
    ser.open()            # 시리얼 포트 열기
    return ser

# 함수 설명: 덱스터러스 핸드 레지스터 쓰기 함수. 매개변수: id는 덱스터러스 핸드 ID, add는 시작 주소, num은 해당 프레임 데이터의 부분 길이, val은 레지스터에 쓸 데이터
def writeRegister(ser, id, add, num, val):
    bytes = [0xEB, 0x90]            # 프레임 헤더
    bytes.append(id)                # id
    bytes.append(num + 3)           # len
    bytes.append(0x12)              # cmd 레지스터 쓰기 명령 플래그
    bytes.append(add & 0xFF)        # 레지스터 시작 주소 하위 8비트
    bytes.append((add >> 8) & 0xFF) # 레지스터 시작 주소 상위 8비트
    for i in range(num):
        bytes.append(val[i])
    checksum = 0x00                 # 체크섬을 0으로 초기화
    for i in range(2, len(bytes)):
        checksum += bytes[i]        # 데이터 합산 처리
    checksum &= 0xFF                # 체크섬의 하위 8비트 취득
    bytes.append(checksum)          # 하위 8비트 체크섬
    
    print("시리얼 포트로 전송한 명령:", [hex(b) for b in bytes])
    
    ser.write(bytes)                # 시리얼 포트에 데이터 쓰기
    time.sleep(0.01)                # 10ms 지연
    ser.read_all()                  # 반환 프레임은 읽어서 버리고 처리하지 않음

# 함수 설명: 덱스터러스 핸드 레지스터 읽기. 매개변수: id는 덱스터러스 핸드 ID, add는 시작 주소, num은 해당 프레임 데이터의 부분 길이, mute는 디버그 플래그
def readRegister(ser, id, add, num, mute=False):
    bytes = [0xEB, 0x90]            # 프레임 헤더
    bytes.append(id)                # id
    bytes.append(0x04)              # len 해당 프레임 데이터 길이
    bytes.append(0x11)              # cmd 레지스터 읽기 명령 플래그
    bytes.append(add & 0xFF)        # 레지스터 시작 주소 하위 8비트
    bytes.append((add >> 8) & 0xFF) # 레지스터 시작 주소 상위 8비트
    bytes.append(num)
    checksum = 0x00                 # 체크섬을 0으로 설정
    for i in range(2, len(bytes)):
        checksum += bytes[i]        # 데이터 합산 처리
    checksum &= 0xFF                # 체크섬의 하위 8비트 취득
    bytes.append(checksum)          # 하위 8비트 체크섬
    
    print("시리얼 포트로 전송한 명령:", [hex(b) for b in bytes])
    
    ser.write(bytes)                # 시리얼 포트에 데이터 쓰기
    time.sleep(0.01)                # 10ms 지연
    recv = ser.read_all()           # 포트에서 바이트 데이터 읽기
    print(recv)
    if len(recv) == 0:              # 반환 데이터 길이가 0이면 바로 반환
        return []
    num = (recv[3] & 0xFF) - 3      # 반환된 레지스터 데이터 개수
    val = []
    for i in range(num):
        value = (recv[7 + i])
        if value > 32767:
            value -= 65536
        val.append(value)
    if not mute:
        print('읽은 레지스터 값은 순서대로: ', end='')
        for i in range(num):
            print(val[i], end=' ')
        print()
    return val

# 함수 기능: 덱스터러스 핸드의 6개 전동 실린더 데이터 쓰기. angleSet은 운동 각도, forceSet은 파지 힘, speedSet은 운동 속도 매개변수 설정
# 매개변수 설명: ID는 덱스터러스 핸드의 해당 ID, str은 설정할 매개변수, val은 설정 데이터
def write6(ser, id, str, val):
    if str == 'angleSet' or str == 'forceSet' or str == 'speedSet' or str == 'mode':
        val_reg = []
        for i in range(6):
            val_reg.append(val[i] & 0xFF)
            val_reg.append((val[i] >> 8) & 0xFF)
        writeRegister(ser, id, regdict[str], 12, val_reg)
    else:
        print('함수 호출 오류. 올바른 사용법: str 값은 \'angleSet\'/\'forceSet\'/\'speedSet\', val은 길이 6의 list이며 값은 0~1000, -1을 자리 표시자로 사용할 수 있습니다.')

# 함수 기능: 덱스터러스 핸드 데이터 읽기
# angleSet은 운동 각도, forceSet은 파지 힘, speedSet은 운동 속도, angleAct는 실제 각도값, forceAct는 각 손가락에 실제로 작용하는 힘
def read6(ser, id, str):
    if str == 'angleSet' or str == 'forceSet' or str == 'speedSet' or str == 'angleAct' or str == 'forceAct' or str == 'temp'  or str == 'errCode'  or str == 'ip' or str == 'statusCode' :
        val = readRegister(ser, id, regdict[str], 12, True) # 읽기
        if len(val) < 12:         # 읽은 데이터가 12보다 작으면 바로 폐기
            print('데이터를 읽지 못했습니다.')
            return
        val_act = []
        for i in range(6):
            value_act = ((val[2*i] & 0xFF) + (val[1 + 2*i] << 8))
            if value_act > 32767:
                value_act -= 65535
            val_act.append(value_act)
        print('읽은 값은 순서대로: ', end='')
        for i in range(6):
            print(val_act[i], end=' ')
        print()
    else:
        print('함수 호출 오류. 올바른 사용법: str 값은 \'angleSet\'/\'forceSet\'/\'speedSet\'/\'angleAct\'/\'forceAct\'/\'errCode\'/\'statusCode\'/\'ip\'입니다.')

def readTouchData(ser):#(법선 힘, 접선 힘, 접선 힘 방향(시계 방향 360도, 접촉이 없으면 65535 반환), 근접 감지(0-16000000))
    # 명령 전송
    cmd = bytes([0xEB, 0x90, 0x01, 0x04, 0x11, 0xB8, 0x0B, 0x44, 0x1D])
    print("촉각 읽기 명령 전송:", [hex(b) for b in cmd])
    ser.write(cmd)
    time.sleep(0.025) # 데이터를 수신하기 위해 25ms 지연
    recv = ser.read_all()
    
    # 원시 응답 출력
    print("원시 응답 데이터:", recv)
    
    # 촉각 데이터 시작 위치(0xB8 0x0B) 찾기
    max_attempts = 5
    for attempt in range(max_attempts):
        try:
            start_idx = recv.index(0xB8)
            if recv[start_idx + 1] == 0x0B:
                data_start = start_idx + 2
                break  # 데이터를 찾았으므로 루프 종료
            else:
                print(f"위치 탐색 시도 {attempt + 1}: 예상한 B8 0B를 찾지 못했습니다.")
        except ValueError:
            print(f"위치 탐색 시도 {attempt + 1}: 촉각 데이터 주소 B8 0B를 찾지 못했습니다.")

        time.sleep(0.01)  # 지연 후 재시도

    else:  # 모든 시도에서 찾지 못하면 None 반환
        print("촉각 데이터 주소 B8 0B를 찾지 못했습니다!!")
        return None, None

    # 다섯 손가락 이름
    fingers = ['little', 'ring', 'middle', 'index', 'thumb']
    finger_results = {}

    for i, finger in enumerate(fingers):
        base_idx = data_start + i * 10  # 각 그룹은 10바이트

        # 10바이트 읽기
        bytes_data = recv[base_idx:base_idx + 10]  

        # 손가락 데이터 조합
        data_bytes = [
            (bytes_data[j] | (bytes_data[j + 1] << 8)) for j in range(0, 6, 2)  # 하위 바이트가 먼저 옴
        ]
    
        # 24비트 근접 감지 데이터
        combined_value = (bytes_data[6] | (bytes_data[7] << 8) | (bytes_data[8] << 16))
        data_bytes.append(combined_value)

        print(f"{finger} 데이터:", data_bytes)

        finger_results[finger] = data_bytes

    # 손바닥 부분의 18바이트를 추출하여 9개 데이터로 구성
    plam_results = {}
    plam_start_idx = data_start + len(fingers) * 10  

    if plam_start_idx + 17 < len(recv):  # 충분한 바이트가 있는지 확인
        plam_data = []
        for j in range(18):
            plam_byte = recv[plam_start_idx + j]
            plam_data.append(plam_byte)

        # 18바이트를 9개 데이터로 조합한다. 2바이트마다 1개 데이터로 조합
        for j in range(9):
            b0 = plam_data[j * 2]      # 하위 바이트
            b1 = plam_data[j * 2 + 1]  # 상위 바이트
            plam_value = (b0 | (b1 << 8))  # 16비트 정수로 조합
            plam_results[f'plam_data_{j + 1}'] = plam_value  # plam_data_1-3(손바닥 왼쪽), plam_data_4-6(손바닥 중앙), plam_data_7-9(손바닥 오른쪽).

    else:
        print("손바닥 데이터가 범위를 벗어나 읽을 수 없습니다.")

    # 다섯 손가락과 손바닥 데이터를 포함한 딕셔너리 반환
    ser.read_all() # 잔여 데이터 지우기
    return finger_results, plam_results

def main():
    # 시리얼 포트 초기화(실제 환경에 맞게 포트와 보드레이트 수정)
    ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=0.1)
    
    print('덱스터러스 핸드 제어 모드 설정, -1은 설정하지 않음을 의미합니다!')
    write6(ser, 1, 'mode', [0, 0, 0, 0, 0, 0]) # ID 번호를 해당 덱스터러스 핸드의 ID로 변경. 0-속도/힘 보호, 1-힘 폐루프, 2-임피던스
    time.sleep(0.1)                   # 0.1초 지연
        
    print('덱스터러스 핸드 운동 속도 매개변수 설정, -1은 해당 운동 속도를 설정하지 않음을 의미합니다!')
    write6(ser, 1, 'speedSet', [4000, 4000, 4000, 4000, 4000, 4000]) # ID 번호를 해당 덱스터러스 핸드의 ID로 변경. val에 대응하는 전동 실린더 ID는 1,2,3,4,5,6; 속도값은 0-4000이며 4000은 최댓값, 0은 움직이지 않음. val을 -1로 설정하면 해당 손가락은 반응하지 않음
    time.sleep(0.1)                   # 0.1초 지연
    print('덱스터러스 핸드 파지 힘 매개변수를 설정합니다!')
    write6(ser, 1, 'forceSet', [6000, 6000, 6000, 6000, 6000, 6000])# ID 번호를 해당 덱스터러스 핸드의 ID로 변경. val에 대응하는 전동 실린더 ID는 1,2,3,4,5,6; 힘 값은 0-12000이며 12000은 최대 힘, 0은 움직이지 않음. val을 -1로 설정하면 해당 손가락은 반응하지 않음
    time.sleep(0.1)                   # 0.1초 지연
    print('덱스터러스 핸드 운동 각도 매개변수 0 설정, -1은 해당 운동 각도를 설정하지 않음을 의미합니다!')
    write6(ser, 1, 'angleSet', [900, 900, 900, 900, 1300, 1700])# ID 번호를 해당 덱스터러스 핸드의 ID로 변경. val에 대응하는 전동 실린더 ID는 1,2,3,4,5,6; 네 손가락의 각도값은 900-1740이며 1740은 최대 각도(174도), 900(90도)은 최소 각도. 엄지 굽힘은 1100-1350이며 1350은 최대 각도(135도), 1100(110도)은 최소 각도. 엄지 측면 벌림은 600-1800이며 1800은 최대 각도(180도), 600(60도)은 최소 각도. -1로 설정하면 해당 손가락은 반응하지 않음
    time.sleep(1)             # 1초 지연
    write6(ser, 1, 'angleSet', [1720, 1720, 1720, 1720, 1350, 1700])# ID 번호를 해당 덱스터러스 핸드의 ID로 변경. val에 대응하는 전동 실린더 ID는 1,2,3,4,5,6; 네 손가락의 각도값은 900-1740이며 1740은 최대 각도(174도), 900(90도)은 최소 각도. 엄지 굽힘은 1100-1350이며 1350은 최대 각도(135도), 1100(110도)은 최소 각도. 엄지 측면 벌림은 600-1800이며 1800은 최대 각도(180도), 600(60도)은 최소 각도. -1로 설정하면 해당 손가락은 반응하지 않음
    time.sleep(1)                   # 1초 지연
    print('덱스터러스 핸드 동작 라이브러리 시퀀스 설정: 1!')
    writeRegister(ser, 1, regdict['actionSeq'], 2, [1,0])   # 1번 시퀀스
    time.sleep(0.1)                   # 0.1초 지연
    print('덱스터러스 핸드의 현재 시퀀스 동작을 실행합니다!')
    writeRegister(ser, 1, regdict['actionRun'], 2, [1,0])   # 실행 플래그 1 쓰기
    time.sleep(1)                   # 1초 지연
    read6(ser, 1, 'forceAct')  
    
    try:
        while True:
            finger_data, plam_data = readTouchData(ser)
            if finger_data is not None and plam_data is not None:
                # 촉각 데이터 출력(법선 힘, 접선 힘, 접선 힘 방향(시계 방향 360도, 접촉이 없으면 65535 반환), 근접 감지(0-16000000))
                print("손가락 데이터:", finger_data)
                print("손바닥 데이터:", plam_data)
            else:
                print("데이터 읽기에 실패했습니다!")

            time.sleep(0.02)  # 50Hz, 20밀리초 간격

    except KeyboardInterrupt:
        print("프로그램 종료")
    finally:
        ser.close()  # 시리얼 포트 닫기

if __name__ == "__main__":
    main()
