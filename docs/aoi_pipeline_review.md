# Rà soát luồng nhận diện lỗi AOI: tách bước, đánh giá, so sánh

> Trạng thái: **baseline**, chưa sửa thuật toán. Số liệu lấy từ `benchmark/results/*_baseline_current.json`.
> Nhãn chuẩn (`benchmark/ground_truth.json`) là **bản nháp**, cần người kiểm vải xác nhận.

## 0. Quy tắc làm việc (chống vá chắp)

1. Không sửa thuật toán khi chưa có benchmark trước và sau:
   `python benchmark/run_benchmark.py --tag before` → sửa → `--tag after` → `python benchmark/compare.py before.json after.json`.
2. Mỗi lần sửa nhắm **một bước, một vấn đề** đã được benchmark chỉ ra (mục 4). Ghi rõ giả thuyết trước khi sửa.
3. `compare.py` trả mã lỗi nếu bất kỳ lỗi bắt buộc nào từng đúng nay sai. **Có hồi quy thì không nhận bản sửa**, dù tổng điểm tăng.
4. Ngưỡng hay hằng số mới phải ghi rõ nguồn gốc: chỉnh trên ảnh nào, đo thế nào.
5. Khi thêm ảnh hoặc loại lỗi mới, **bổ sung ground truth trước**, rồi mới sửa code.

## 1. Sơ đồ luồng

```
Ảnh RGB 2048x2448
 │
 S0 Tiền xử lý ── (tuỳ chọn) Auto-Crop → thu nhỏ ≤1024 px → nền Gaussian 4.5% bề rộng → nhiễu vân (median độ lệch chuẩn cục bộ)
 │
 S1 Bản đồ dò (mỗi bản đồ 0..1, mức quyết định do chế độ chọn)
 │    D1 Màu (chromatic)  : ΔE Lab so với nền cục bộ + lệch màu với nền trung vị rộng (bỏ khối chạm mép khung)
 │    D2 Nếp (crease)     : Hessian đường mảnh (12 hướng) ⊕ bộ dò nếp mềm (lọc định hướng + lấy trung bình dọc 81/161/241 px)
 │    D3 Tương phản       : |xám − nền| vượt sàn nhiễu, mở hình thái 5x5
 │    D4 Mẫu Vàng (tuỳ chọn): PatchCore, khoảng cách tới bank vải tốt
 │    D5 CRAS (chỉ chế độ AI): mô hình ngoài miền dữ liệu → không dùng
 │
 S2 Vùng kiểm tra ── ROI cột ± biên vải (1 in) → xoá bản đồ ngoài vùng
 │
 S3 Tách lỗi (mỗi bản đồ riêng ở chế độ Hỗn Hợp) ── ngưỡng tuyệt đối → nối nét 0.31 in (đóng ngang/dọc) → thành phần liên thông → lọc diện tích ≥150 px
 │
 S4 Phân loại (cây luật) ── Δb vàng → Stain | dạng đường → ChalkMark (tương phản >45) / Crease | khối sáng đặc → Hole (chờ xác nhận) | còn lại → Weave
 │
 S5 Gộp trùng giữa các bản đồ ── chồng lấn >50% → giữ loại ưu tiên (Hole > Stain > Chalk/Crease > Weave)
 │
 S6 Điểm ASTM ── chiều dài hộp xoay / px_per_inch → 1–4 điểm; Hole chờ xác nhận không tính
 │
 S7 Thống kê khung / tích luỹ cuộn ── ≤4 điểm/yard, điểm/100 yd², hạng theo ngưỡng buyer
```

Chế độ trong app: **Hỗn Hợp** = D1+D2+D3 (+D4), tách theo từng bản đồ, ngưỡng 0.40 · **Nếp Gấp** = D2, ngưỡng 0.18 ·
**Vết Ố** = D1, ngưỡng 0.45 · **Mẫu Vàng** = D4, ngưỡng 0.50 · **AI Sâu** = D5, ngưỡng 0.50.

## 2. Chi tiết từng bước: tham số, nguồn gốc, giả định

| Bước | Tham số chính | Nguồn gốc | Giả định | Rủi ro đã biết |
|---|---|---|---|---|
| S0 | thu nhỏ 1024, nền = Gaussian 4.5% bề rộng | Code gốc | Vải đồng màu, ánh sáng biến thiên chậm | Vết lớn hơn cửa sổ nền chỉ còn phần viền |
| D1 | sàn = max(1.8·nhiễu, 6) ΔE; vàng Δb−6, /20; nền màu trung vị 60% | Code gốc; phần nền trung vị do Claude thêm, chỉnh trên ảnh tổng hợp + 154829 | Vết ố khác màu nền | Viền vải / mặt bàn tạo phản hồi màu |
| D2a | Hessian σ≈2, mở theo đường 3.5% bề rộng ×12 hướng, sàn max(4.5·nhiễu, 18)/45 | Code gốc (chỉ ngang/dọc); 12 hướng do Claude thêm | Nếp là đường mảnh, tương phản rõ | Không thấy nếp mềm, rộng |
| D2b | FOLD_LEVELS {81: 7.8, 161: 6.0, 241: 4.9}, σ ngang 4 | **Claude chỉnh trên vùng sạch của 1 ảnh (154200)** | Nếp thẳng, dài vài inch | Vân vải tạo phản hồi giả; phản ứng với cạnh (viền vải) |
| D3 | sàn max(5·nhiễu, 28)/50 | Code gốc | Lỗi tương phản mạnh | Gần như "chết" trên vải đen (phản hồi 0.00) |
| S3 | ngưỡng theo chế độ; nối 0.31 in; ≥150 px | Code gốc (quy đổi ra inch) | 1 lỗi = 1 thành phần liên thông | Nối nét gộp lỗi lân cận; nếp đứt đoạn thành nhiều mảnh |
| S4 | Δb >12 (hoặc >6 nếu nền sáng); AR ≥2.8; tương phản >45; luật Hole | Code gốc; Hole do Claude viết | Điểm ảnh của khối đại diện cho lỗi | Khối rộng hơn lỗi thì màu/sáng bị pha loãng |
| S5 | chồng lấn >50%, bảng ưu tiên | Claude | — | Chưa kiểm chứng rộng |
| S6/S7 | ASTM 3/6/9 in; lỗ ≤1 in; px/inch = 100 | Tiêu chuẩn; **px/inch chưa đo thật** | — | Sai tỉ lệ thì sai điểm |

## 3. Kết quả baseline (5 ảnh cuộn, 20 lỗi bắt buộc)

| Chế độ | Tìm thấy | Đúng loại | Báo nhầm | Lỗi hỏng ở bước |
|---|---|---|---|---|
| Hỗn Hợp | 12/20 | 10/20 | 2 | S1 bản đồ: 6 · S2 tách lỗi: 2 · S3 phân loại: 2 |
| Nếp Gấp | 12/20 | 9/20 | 4 | S1: 5 · S2: 3 · S3: 3 |
| Vết Ố | 5/20 | 5/20 | 0 | S1: 15 (đúng thiết kế: chế độ này không dò nếp) |

Theo loại lỗi (chế độ Hỗn Hợp):
- **Nếp:** 7/15 đúng.
- **Vết ố:** 1/1.
- **Chấm:** 1/1.
- **Vết phấn/vết lạ:** tìm thấy 3/3, nhưng **chỉ 1/3 đúng loại**.

Chẩn đoán từng lỗi bắt buộc chưa đạt (chế độ Hỗn Hợp, độ mạnh p95 trên bản đồ màu / nếp / tương phản):

| Lỗi | Hỏng ở | Bằng chứng |
|---|---|---|
| 154200-F1 (nếp ngang chữ T) | S1 | Bản đồ nếp 0.30 < ngưỡng 0.40. Chế độ Nếp Gấp (0.18) thấy nhưng chỉ phủ < 50% chiều dài |
| 154231/250-F3 (nếp dọc mờ) | S1 | Bản đồ nếp 0.00 |
| 154231/250-F4 (dải chéo rộng) | S1 | Bản đồ nếp 0.09–0.11 |
| 154231/250-F5 (nếp chéo trái) | S2 | Bản đồ nếp 0.98, nhưng lỗi tách ra phủ < 50% chiều dài |
| 154325-F3 (nếp ngang dưới phải) | S1 | Bản đồ nếp 0.01 |
| 154325-M1/M2 (vạch phấn) | S3 | Tìm thấy nhưng xếp thành **Crease**. Chế độ Vết Ố xếp đúng ChalkMark |

## 4. Các vấn đề theo bước, xếp ưu tiên (chưa sửa; mỗi mục cần giả thuyết + benchmark)

**P1 · S1-D2: bản đồ nếp không tách được nếp khỏi nền.** Phản hồi nền (p99.9 ngoài vùng nhãn) của bản đồ nếp
**= 1.00** trên 4/5 ảnh, tức nền còn mạnh ngang hoặc hơn nếp thật. Đây là nguồn gốc của cả việc sót (S1) lẫn
việc lan ra ngoài lỗi (có ảnh tới 68%).
Liên quan: POC `poc/single_image_fold` (trên ảnh phẳng, ngưỡng nào cũng không vừa bắt đủ vừa ít báo nhầm) và
POC `poc/photometric_stereo` (giải pháp chiếu sáng). **Quyết định phần cứng nên đi trước việc chỉnh D2.**

**P2 · S4: phân loại dựa trên trung bình điểm ảnh của khối.** Vạch phấn trắng trên vải đen bị xếp thành Crease,
vì khối trên bản đồ nếp rộng hơn nét phấn nên độ sáng trung bình bị pha loãng (tương phản < 45). Lỗi cùng họ với
lỗi "vết ố bị nuốt" đã sửa ở S5. Hướng cần đánh giá: đo độ tương phản / màu trên các **điểm ảnh lõi** của khối
(ví dụ phân vị cao), hoặc phân loại theo bản đồ đã phát hiện ra khối.

**P3 · S3: tách lỗi cho nếp đứt đoạn.** 154231-F5: bản đồ thấy rõ (0.98) nhưng khối tách ra phủ < 50% chiều dài.
Cần xem nối nét (0.31 in, chỉ ngang/dọc) có phù hợp với nếp chéo không.

**P4 · S2/cấu hình: ROI mặc định = toàn khung.** Dải nền trắng ở mép (154231/154250) tạo một "Crease" dài 24 in.
Đây là việc cấu hình tại trạm (đặt ROI), không phải việc của thuật toán. Benchmark chạy với cấu hình mặc định
nên con số "lan" (sprawl) trên 2 ảnh đó bị đội lên.

**P5 · D3 tương phản gần như vô dụng trên vải đen** (phản hồi 0.00). Cần đánh giá xem nên giữ, bỏ hay thay.

## 4b. Nhật ký thử nghiệm (2026-09-26): mục tiêu P2, vạch phấn 154325-M1/M2 bị xếp thành Crease

| # | Giả thuyết | Cách thử | Kết quả | Quyết định |
|---|---|---|---|---|
| H1 | Tương phản bị pha loãng do lấy trung bình cả khối → đo trên lõi khối | Đo tương phản trung bình và lõi p80/p90 trên các khối (không sửa code) | Vạch phấn có giá trị **giống hệt** nếp F2/F3 → cùng một khối gộp. Đo lõi cũng chỉ ra 19–22, dưới 45 | **Bác bỏ**: nguyên nhân là gộp khối, không phải cách đo |
| — | Khối gộp đến từ đâu? | Đo độ rộng phản hồi quanh vạch phấn; tô nét phấn thành màu vải rồi đo lại | Dải rộng ~250 px đến từ **bộ dò nếp mềm (D2b)** và **vẫn còn khi không có nét phấn** → hiệu ứng mép ảnh: dốc sáng ở mép + đệm biên kiểu phản chiếu tạo "rãnh chữ V" giả | Xác định được nguyên nhân 1 |
| H2 | Chặn biên đầu vào D2b ở ±3σ | Sửa code + benchmark | Vạch phấn vẫn sai; **2 hồi quy** (chế độ Nếp Gấp) | **Loại**, đã hoàn tác |
| H3 | Khử dốc sáng + đệm biên bằng 0 cho D2b | Sửa code + benchmark | **6 hồi quy**, nếp bắt được giảm | **Loại**, đã hoàn tác. Ngưỡng `FOLD_LEVELS` gắn chặt với cách tiền xử lý cũ |
| E1 | D2b đóng góp bao nhiêu? | Tắt D2b trong bộ nhớ (`benchmark/experiment.py`) | Nếp: 7/15 → **0/15**. Bộ dò đường mảnh D2a không bắt được nếp nào đã gán nhãn | Không thể bỏ D2b |
| E2 | Chỉ tin D2b ở vùng kernel lọt hẳn trong ảnh | Thay hàm trong bộ nhớ | **2 hồi quy**, thêm báo nhầm; vạch phấn **vẫn** sai | **Loại**. Có nguyên nhân 2: D2a cũng tạo khối rộng hơn nét phấn |

**Kết luận của đợt này:**
1. Lỗi vạch phấn **không sửa riêng được ở bước phân loại**. Nó là tổ hợp của: (a) D2b phản hồi giả ở mép ảnh,
   (b) D2a tạo khối rộng hơn nét, (c) phân loại theo trung bình khối, (d) luật gộp trùng khi hai khối cùng ưu tiên
   thì giữ khối lớn hơn.
2. **Mọi khả năng bắt nếp đều dựa vào D2b**, một bộ dò được chỉnh tay trên 1 ảnh. Hễ thay đổi cho đúng nguyên lý
   (H3) là mất hiệu chuẩn. Đây là nền móng yếu nhất của hệ thống, và vướng vào quyết định chiếu sáng (P1).
3. Code sản xuất giữ nguyên như baseline (đã kiểm tra lại bằng `revert_check2`: 0 hồi quy).

## 5. Giới hạn của bộ đánh giá hiện tại

- **Chỉ có 6 ảnh.** 154231 và 154250 gần như trùng nhau; 155041 là thân vải cắt nên báo cáo riêng.
  Số liệu dùng để **định hướng và chặn hồi quy**, chưa đủ để kết luận thống kê.
- **Nhãn là bản nháp.** Một số "báo nhầm" hoặc "lan" có thể là nếp thật chưa được gán nhãn.
- **Chưa có khung vải sạch** (không lỗi), nên chưa đo được tỷ lệ báo nhầm trên vải tốt, là chỉ số quan trọng nhất
  khi chạy thật.
- Cần bổ sung: 10–20 khung sạch và 10–20 khung có lỗi cho mỗi loại vải; nhãn do người kiểm vải xác nhận.

## 6. Cách chạy

```bash
python benchmark/run_benchmark.py --tag <ten_lan_chay>
python benchmark/compare.py benchmark/results/<A>.json benchmark/results/<B>.json
```

Ảnh soát nhãn: `benchmark/label_review/*.jpg` (xanh = bắt buộc, vàng = không chắc, xám = vùng bỏ qua).
