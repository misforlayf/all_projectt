CREATE DATABASE market;
USE market;

CREATE TABLE urunler(
urun_no INT IDENTITY(1,1) PRIMARY KEY,
urun_ad VARCHAR(20) UNIQUE NOT NULL,
urun_mýktarý INT NULL,
);

INSERT INTO urunler VALUES 
('makarna', '25');

