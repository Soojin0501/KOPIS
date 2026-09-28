-- KOPIS 공연 데이터 관계형 모델
-- 원천은 시설·공연장·공연·회차·거래가 한 행에 반복되는 71열 파일이다.
-- 01_profile_raw.py 와 02_check_keys.py 로 확인한 종속 관계대로 나눈다.

-- ───────── 기준 정보 ─────────

CREATE TABLE operator (                 -- 전송사업자 (예매처)
    operator_code   VARCHAR PRIMARY KEY,
    operator_name   VARCHAR NOT NULL
);

CREATE TABLE facility (                 -- 공연시설 (건물)
    facility_code   VARCHAR PRIMARY KEY,
    facility_name   VARCHAR NOT NULL,
    facility_type   VARCHAR,            -- 시설특성
    open_year       INTEGER,
    address         VARCHAR,
    region          VARCHAR,            -- 공연지역명. 시설에 종속됨을 확인
    has_restaurant  BOOLEAN, has_cafe BOOLEAN, has_store BOOLEAN,
    has_playroom    BOOLEAN, has_nursing_room BOOLEAN,
    acc_parking     BOOLEAN, acc_restroom BOOLEAN, acc_ramp BOOLEAN, acc_elevator BOOLEAN,
    parking_own     BOOLEAN, parking_public BOOLEAN
);

CREATE TABLE hall (                     -- 공연장 (시설 안의 홀). 시설 1 : 공연장 N
    hall_code       VARCHAR PRIMARY KEY,
    facility_code   VARCHAR NOT NULL REFERENCES facility(facility_code),
    hall_name       VARCHAR NOT NULL,
    seats           INTEGER,
    accessible_seats INTEGER,
    has_orchestra_pit BOOLEAN, has_practice_room BOOLEAN, has_dressing_room BOOLEAN,
    stage_size      VARCHAR
);

CREATE TABLE genre (                    -- 장르 (중분류)
    genre_id        INTEGER PRIMARY KEY,
    genre_name      VARCHAR NOT NULL UNIQUE,
    is_composite    BOOLEAN NOT NULL    -- '복합' 은 실제 장르가 아니라 "미분류 조합" 표시
);

CREATE TABLE subgenre (                 -- 세부장르. 이름만으로는 유일하지 않다 (기악·성악이 두 장르에 있음)
    subgenre_id     INTEGER PRIMARY KEY,
    genre_id        INTEGER NOT NULL REFERENCES genre(genre_id),
    subgenre_name   VARCHAR NOT NULL,
    UNIQUE (genre_id, subgenre_name)
);

CREATE TABLE booking_channel (channel_code VARCHAR PRIMARY KEY, channel_name VARCHAR NOT NULL);
CREATE TABLE payment_method  (payment_code VARCHAR PRIMARY KEY, payment_name VARCHAR NOT NULL);
CREATE TABLE discount_type   (discount_code VARCHAR PRIMARY KEY, discount_name VARCHAR NOT NULL);

-- ───────── 공연 ─────────

CREATE TABLE performance (              -- 공연
    performance_id  INTEGER PRIMARY KEY,   -- 대리키. 가공 파일에는 공연코드가 없다
    performance_code VARCHAR UNIQUE,       -- 원천의 자연키. 없으면 NULL
    title           VARCHAR NOT NULL,      -- 유일하지 않다. 키로 쓰지 않는다
    hall_code       VARCHAR REFERENCES hall(hall_code),
    subgenre_id     INTEGER REFERENCES subgenre(subgenre_id),
    start_date      DATE, end_date DATE,
    running_time    VARCHAR, age_limit VARCHAR,
    is_child BOOLEAN, is_festival BOOLEAN, is_visiting BOOLEAN, is_open_run BOOLEAN,
    source          VARCHAR NOT NULL       -- raw_sample | single | composite
);

CREATE TABLE performance_genre (        -- 공연 N : 장르 M. 복합 공연을 장르 조합으로 푼다
    performance_id  INTEGER NOT NULL REFERENCES performance(performance_id),
    genre_id        INTEGER NOT NULL REFERENCES genre(genre_id),
    basis           VARCHAR NOT NULL,      -- label(전산망 기록) | predicted(모델 추정)
    rank_no         INTEGER NOT NULL,      -- 1 = 주 장르, 2 = 둘째 장르
    probability     DOUBLE,                -- 추정일 때만
    PRIMARY KEY (performance_id, genre_id, basis)
);

-- ───────── 사람과 단체 ─────────

CREATE TABLE person (                   -- 인물. 고유 식별자가 없어 준식별자를 쓴다
    person_id       INTEGER PRIMARY KEY,
    person_name     VARCHAR NOT NULL,
    genre_id        INTEGER NOT NULL REFERENCES genre(genre_id),  -- 동명이인을 가르는 기준
    UNIQUE (person_name, genre_id)
);

CREATE TABLE organization (             -- 기획·제작 단체
    org_id          INTEGER PRIMARY KEY,
    org_name        VARCHAR NOT NULL UNIQUE
);

CREATE TABLE performance_person (       -- 공연 N : 인물 M. 역할이 관계의 속성이다
    performance_id  INTEGER NOT NULL REFERENCES performance(performance_id),
    person_id       INTEGER NOT NULL REFERENCES person(person_id),
    role            VARCHAR NOT NULL,      -- cast(출연) | crew(제작)
    PRIMARY KEY (performance_id, person_id, role)
);

CREATE TABLE performance_org (          -- 공연 N : 단체 M
    performance_id  INTEGER NOT NULL REFERENCES performance(performance_id),
    org_id          INTEGER NOT NULL REFERENCES organization(org_id),
    role            VARCHAR NOT NULL,      -- 주최 | 주관 | 제작사 | 기획사 | 미상
    PRIMARY KEY (performance_id, org_id, role)
);

-- ───────── 회차와 거래 ─────────

CREATE TABLE show_session (             -- 회차. 원천의 공연회차 열은 전부 1 이라 쓸 수 없다
    session_id      INTEGER PRIMARY KEY,
    performance_id  INTEGER NOT NULL REFERENCES performance(performance_id),
    show_at         TIMESTAMP NOT NULL,
    UNIQUE (performance_id, show_at)
);

CREATE TABLE ticket_event (             -- 예매·취소 거래. 입장권 하나에 예매와 취소가 따로 붙는다
    event_id        BIGINT PRIMARY KEY,    -- 대리키. 입장권번호+구분 조합도 유일하지 않았다
    ticket_no       VARCHAR NOT NULL,
    session_id      INTEGER NOT NULL REFERENCES show_session(session_id),
    operator_code   VARCHAR NOT NULL REFERENCES operator(operator_code),
    event_type      VARCHAR NOT NULL,      -- booking | cancel
    event_at        TIMESTAMP NOT NULL,
    quantity        INTEGER NOT NULL,
    amount          DOUBLE NOT NULL,
    unit_price      DOUBLE,
    discount_amount DOUBLE,
    channel_code    VARCHAR REFERENCES booking_channel(channel_code),
    payment_code    VARCHAR REFERENCES payment_method(payment_code),
    discount_code   VARCHAR REFERENCES discount_type(discount_code),
    gender          VARCHAR,
    age             INTEGER
);
