# POC · Photometric Stereo cho nếp gấp vải

Thử nghiệm **độc lập** với hệ thống chính (`app.py`): không import code sản xuất, chạy app riêng ở cổng riêng.
Mục tiêu duy nhất: **đo bằng số liệu** xem chiếu sáng 4 hướng xiên có làm nếp gấp nổi rõ hơn đèn hiện tại không,
trước khi quyết định phần cứng.

## Vì sao cần POC này

Trên ảnh chụp bằng đèn hiện tại, nếp gấp trên vải đen chỉ lệch khoảng 3 mức xám. Trong khi đó, vân loang tự nhiên
của vải dao động khoảng 5 mức xám, nên SNR chỉ khoảng 0.55: nếp chìm dưới vân. Không thuật toán 2D nào tách ổn định
được "bóng của nếp" với "vân màu của vải".

Nếp gấp là **hình dạng 3D**, còn vân là **màu (albedo)**. Khi chụp dưới đèn từ trái và từ phải, tỷ số
`(I_phải − I_trái) / (I_phải + I_trái)` triệt tiêu albedo (theo mô hình Lambert) và chỉ còn độ dốc bề mặt.
Làm tương tự với cặp trên/dưới sẽ ra ảnh hình dạng (Laplacian của độ cao), trên đó nếp gấp nổi lên.
Đây là nguyên lý của Keyence LumiTrax, Basler photometric stereo và toán tử `photometric_stereo` của HALCON.

## Quy trình chụp (thí nghiệm 1–2 giờ)

1. Cố định camera và tấm vải; không chạm vào vải giữa các lần chụp.
2. Tắt đèn trần hoặc che bớt ánh sáng xung quanh.
3. Đặt một thanh LED hoặc đèn pin mạnh **thấp khoảng 15–20° so với mặt vải**, cách vùng chụp khoảng 50 cm.
4. Chụp **4 ảnh**: đèn đặt ở phía **TRÁI**, **PHẢI**, **TRÊN**, **DƯỚI** của khung hình (xét theo ảnh).
5. Chụp thêm **1 ảnh bằng đèn hiện tại** để so sánh.
6. **Khóa phơi sáng và gain thủ công**, giống nhau cho cả 4 ảnh xiên. Không để ảnh bị cháy sáng (bão hòa).
7. Lưu PNG/TIFF gốc (không nén JPEG nếu được; ảnh 12/16-bit dùng được).
8. Làm thêm 1 bộ trên vải trắng nếu có.

## Chạy

```bash
# App giao diện riêng (cổng 8503)
streamlit run poc/photometric_stereo/app_ps.py --server.port 8503

# Dòng lệnh: tự kiểm tra bằng dữ liệu tổng hợp
python poc/photometric_stereo/run_ps.py --synthetic --out out_synth

# Dòng lệnh: ảnh thật + các nếp cần đo
python poc/photometric_stereo/run_ps.py --left L.png --right R.png --top T.png --bottom B.png \
       --front F.png --lines lines.json --elevation 18 --out out_real

# Test
python poc/photometric_stereo/test_ps.py
```

`lines.json`: `[{"name": "1 nep ngang", "seg": [x0, y0, x1, y1]}, ...]` (pixel của ảnh gốc).

## Đọc kết quả

Bảng SNR được đo **cùng một cách** cho 3 loại ảnh: đèn hiện tại, 1 đèn xiên, và ảnh hình dạng photometric stereo (4 đèn).

| SNR mỗi vị trí | Ý nghĩa |
|---|---|
| < 1 | Nếp chìm trong nhiễu; thuật toán nào cũng phải đoán |
| 2–3 | Nhìn thấy được, dò được nhưng vẫn cần chỉnh |
| > 5 | Rõ ràng; một bộ dò đơn giản, không cần chỉnh tay, là đủ |

Kết quả trên dữ liệu tổng hợp (mô hình Lambert lý tưởng, **chỉ để kiểm tra code**):
đèn thẳng cho SNR 0.16–0.39, 1 đèn xiên cho 0.3–1.7, 4 đèn cho 17–45.
Vải thật không lý tưởng (có bóng sợi, tự đổ bóng), nên con số thật sẽ thấp hơn. **Chỉ ảnh chụp thật mới là phép thử.**

## Các file

| File | Vai trò |
|---|---|
| `ps_core.py` | Tính photometric stereo: gradient từ tỷ số cặp đèn đối diện, ảnh shape, albedo và normal (Woodham), căn chỉnh ảnh |
| `snr.py` | Đo SNR của nếp trên bất kỳ ảnh nào, cùng một phương pháp |
| `synthetic.py` | Bộ ảnh tổng hợp: vải đen có vân loang + nếp đã biết vị trí |
| `run_ps.py` | Công cụ dòng lệnh |
| `app_ps.py` | App Streamlit riêng cho POC |
| `test_ps.py` | Test: albedo bị triệt tiêu, dấu/độ lớn của gradient, PS tốt hơn đèn thẳng, căn chỉnh, SNR trên nhiễu thuần |

## Ngoài phạm vi POC (tính sau khi có kết quả)

Đồng bộ strobe với camera quét dòng hoặc camera vùng trên máy kiểm cuộn, vải di chuyển giữa các lần chớp đèn,
hiệu chuẩn hướng đèn, tốc độ xử lý, và tích hợp vào `app.py`.
