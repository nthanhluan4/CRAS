# POC · Dò nếp gấp trên MỘT ảnh phẳng (đèn hiện tại)

Thử nghiệm độc lập với `app.py`. Mục đích: trả lời bằng số liệu câu hỏi **"chỉ với 1 ảnh phẳng, có nhận diện được
nếp gấp không, và tốt tới đâu?"**

## Nội dung

| File | Vai trò |
|---|---|
| `labels.json` | Nhãn nếp gấp **nháp do Claude đánh** trên 4 ảnh vải đen. `required` = bắt buộc bắt được, `optional` = không chắc, `ignore` = vùng bỏ qua (dải nền trắng, đường phấn). **Cần người kiểm vải xác nhận hoặc sửa.** |
| `label_review/*.jpg` | Ảnh tăng cường tương phản có vẽ nhãn, để soát lại nhãn |
| `baseline_current.py` | **Bản sao** bộ dò nếp đang chạy trong `app.py`, dùng làm mốc so sánh |
| `detector_mf.py` | Bộ dò mới: làm trắng nhiễu vân vải, lọc khớp định hướng (bề rộng × độ dài × góc), ngưỡng a-contrario |
| `evaluate.py` | Chạy các bộ dò, xuất ảnh chồng kết quả và thống kê theo từng nếp |
| `sweep.py` | Quét ngưỡng: bắt được / độ chính xác / diện tích báo nhầm. **Đây là thước đo chính.** |

```bash
python poc/single_image_fold/sweep.py
python poc/single_image_fold/evaluate.py --out out_eval
```

## Kết quả (4 ảnh, 15 nếp required, nhãn nháp)

**Bộ dò mới (lọc khớp, `mf_z`):**

| Ngưỡng z | Bắt được | Độ chính xác | Diện tích báo nhầm |
|---|---|---|---|
| 5 | 15/15 | 25% | 11.6% |
| 6 | 13/15 | 39% | 3.8% |
| 7 | 9/15 | 47% | 1.7% |
| 8 | 9/15 | 52% | 1.0% |
| 10 | 5/15 | 60% | 0.3% |

**Bộ dò hiện tại (`current`):** tốt nhất bắt được 8/15 nếp, độ chính xác khoảng 25%.

## Kết luận

1. **Bộ dò mới tốt hơn bộ dò hiện tại** ở mọi điểm vận hành.
2. **Không có ngưỡng nào vừa bắt đủ vừa ít báo nhầm.** Vân của vải đen tự tạo ra các cấu trúc dạng đường dài,
   mạnh ngang với nếp mềm (z khoảng 5–7). Nếp mạnh có z khoảng 10–12.
3. **Ngưỡng a-contrario lý thuyết (z ≈ 5.8) không đạt mục tiêu** ≤ 1 báo nhầm/ảnh, vì vân vải sau khi làm trắng
   không phải nhiễu Gaussian độc lập. Muốn ngưỡng có cơ sở thì phải **đo phân bố nền thực tế trên vải sạch**
   (khung vải không có nếp), không dùng giả định lý thuyết.
4. **Dùng được ngay:** ở ngưỡng cao (z ≥ 10), bộ dò bắt được các **nếp mạnh** với báo nhầm rất ít (0.3% diện tích).
   **Nếp mềm trên 1 ảnh phẳng thì chưa đạt yêu cầu sản xuất.**

## Lưu ý về độ tin cậy

- Nhãn là **nháp**. Một phần "báo nhầm" có thể là nếp thật chưa được đánh nhãn, nhất là trên 154325. Nếu vậy
  thì độ chính xác thật cao hơn con số trên.
- Mới có 4 ảnh (và 154231 gần như trùng 154250), nên số liệu mang tính định hướng, chưa có ý nghĩa thống kê.

## Bước tiếp theo cho hướng ảnh phẳng

1. Người kiểm vải **soát và sửa `labels.json`** dựa trên `label_review/`.
2. Chụp thêm **10–20 khung vải sạch** (không nếp) và **10–20 khung có nếp** cùng loại vải, cùng điều kiện đèn.
3. Dùng khung sạch để **đo phân bố z của nền**, từ đó đặt ngưỡng theo mục tiêu báo nhầm (ví dụ 1 lần mỗi 100 m).
4. Chạy lại `sweep.py` để có đường recall–precision đáng tin cậy, rồi quyết định: ảnh phẳng chỉ bắt nếp mạnh,
   hay chuyển sang chiếu sáng xiên (POC `../photometric_stereo`).
