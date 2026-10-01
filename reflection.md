# Day 14 — Reflection

## Evaluation Report & Failure Analysis

Dùng kết quả thật trong `artifacts/benchmark_results.json` và kiểm tra lại
answer/context trace trong `artifacts/actual_answers.json` trước khi kết luận.

*Run: generator `gpt-4o-mini`, BM25 retriever (paragraph chunks), top_k = 5,
20 golden QA (5 Easy + 7 Medium + 5 Hard + 3 Adversarial).*

---

## 1. Benchmark Results Summary

**Overall pass rate:** 35.0% (7/20)

| Metric | Average | Min | Max | Nhận xét |
|---|---:|---:|---:|---|
| Context Recall | 0.772 | 0.333 (A01) | 1.000 (E01) | Easy gần như đủ evidence; thấp ở case dùng từ khác corpus (M06 "hacked") và A01/A03 không retrieve được `00_system_scope.md` (A02 có, ở rank 1). |
| Context Precision | 0.904 | 0.325 (M06) | 1.000 (E01) | Metric tốt nhất: khi chunk liên quan được retrieve thì thường đứng đầu. Ngoại lệ là M06, chunk nhiễu chiếm rank 1–3. |
| Faithfulness | 0.544 | 0.100 (A01) | 0.857 (E04) | Đo overlap với **gold context**, nên answer dùng chunk đúng khác (E02) hoặc câu từ chối ngắn (A01) bị phạt; nhưng cũng bắt được A03 lệch khỏi evidence. |
| Relevance | 0.511 | 0.333 (A02) | 0.800 (M04) | Thấp chủ yếu vì question dài và answer paraphrase; là nguyên nhân chính của 7 nhãn `off_topic`. |
| Completeness | 0.503 | 0.100 (A01) | 0.957 (E01) | Metric yếu nhất có ý nghĩa thật: Hard/Medium thiếu điều kiện, phí, deadline (H02, H03, M03, M07). |
| Overall Score | 0.519 | 0.221 (A01) | 0.767 (E05) | Không case nào đạt mức Good; Easy tốt nhất, Hard và Adversarial kém nhất. |

**Score interpretation**

- Metrics/cases ở mức Good (0.8–1.0): Context Precision (0.904). Không case nào có Overall ≥ 0.8.
- Metrics/cases ở mức Needs Work (0.6–0.8): Context Recall (0.772); 7 case có Overall 0.6–0.8: E01, E03, E04, E05, M04, M05, H05.
- Metrics/cases ở mức Significant Issues (<0.6): Faithfulness, Relevance, Completeness, Overall; 13 case còn lại, toàn bộ Hard trừ H05 và toàn bộ Adversarial.

**Failure type distribution** (13 failures / 20 cases)

| Failure Type | Count | Percentage |
|---|---:|---:|
| hallucination | 3 | 23.1% |
| irrelevant | 0 | 0% |
| incomplete | 3 | 23.1% |
| off_topic | 7 | 53.8% |
| refusal | 0 | 0% |

**Chẩn đoán tổng quan:** Vấn đề chính nằm ở retrieval, generation hay cả hai?
Dùng ít nhất hai metrics để bảo vệ kết luận.

> *Câu trả lời:* Vấn đề chính nằm ở **generation**, retrieval là vấn đề phụ ở một số case cụ thể.
> - **Retrieval nhìn chung ổn:** Context Precision 0.904 và Context Recall 0.772. Ở H01 (Recall 0.829, Precision 1.000), chunk quyết định `OT-09-P04` đứng rank 1 nhưng answer vẫn sai.
> - **Generation yếu:** Faithfulness 0.544 và Completeness 0.503 thấp hơn hẳn retrieval. Có 3 case trả lời **sai fact** dù đã có evidence: H01 nói 45 ngày thay vì 21, H03 hứa có loaner cho repair không được bảo hành, A03 chấp nhận premise "lifetime warranty".
> - **Retrieval là nguyên nhân chính ở một số case:** M06 (Recall 0.429, Precision 0.325, chunk `OT-08-P02` chỉ đứng rank 13). Các case adversarial cũng không luôn retrieve được `00_system_scope.md` (A01 và A03 không có chunk OT-00 nào trong top-5).
> - **Một phần failure là lỗi của metric:** E04 trả lời đúng hoàn toàn nhưng fail vì Relevance 0.444; A02 từ chối đúng nhưng bị gán `off_topic`. Vì vậy 7 nhãn `off_topic` không nên đọc là 7 lỗi thật.

---

## 2. Top 3 Worst Failures — 5 Whys

Phân loại failure trước khi đề xuất fix. Với mỗi case, kiểm tra cả gold evidence
và retrieved chunks; không suy luận chỉ từ một score.

*Cách chọn:* ba failure dưới đây được chọn theo **mức độ nghiêm trọng thực tế**
và để đại diện ba nguyên nhân khác nhau (generation sai policy, retrieval miss,
premise sai), không chỉ theo Overall thấp nhất. A01 có Overall thấp nhất
(0.221) nhưng hành vi về cơ bản đúng — nó là false positive của metric và được
xử lý trong cluster 4. H02 (0.331) đúng kết luận, chỉ thiếu chi tiết.

### Failure 1

**ID và question:**

> **H01** — "I was an OrbitPlus member when I ordered a PulsePhone X on August 28, 2026. It was delivered on September 3 and is still unopened. How many days do I have to return it?"

**Expected answer:**

> Return Policy version 1.0 applies because the order was placed before September 1, 2026; the order-placement date decides the version, not the delivery date. Version 1.0 allows 21 calendar days for unopened devices, counted from confirmed delivery on September 3. The 45-day OrbitPlus unopened-device benefit does not apply, because it was introduced with version 2.0 and pre-September 1 orders keep the 21-day window regardless of membership.

**Actual answer:**

> You have 45 days to return the unopened PulsePhone X, as you were an OrbitPlus member when you ordered it. The return window extends from the standard 30 days to 45 days for eligible purchases made while membership is active. The delivery date was September 3, 2026, so you can return it until October 18, 2026.

**Scores:** Context Recall: 0.829 | Context Precision: 1.000 | Faithfulness: 0.375 |
Relevance: 0.619 | Completeness: 0.293 | Overall: 0.429

**Evidence inspection:** Retriever lấy đúng/thiếu/thừa chunks nào?

> Retrieved theo thứ tự: `OT-09-P04`, `OT-05-P01`, `OT-03-P05`, `OT-06-P01`, `OT-01-P02`. Chunk quyết định `OT-09-P04` (Return Policy v1.0 cho order trước 01/09, 21 ngày, "regardless of membership") đứng **rank 1**, nên retrieval **không thiếu**. Nhưng top-5 cũng chứa hai chunk của policy hiện hành: `OT-05-P01` (v2.0, 30 ngày) và `OT-03-P05` (OrbitPlus mở rộng 30 → 45 ngày). Model chọn theo hai chunk này. Answer sai kết luận (45 thay vì 21 ngày, hạn 18/10 thay vì 24/09), nhưng metric chỉ gán `incomplete`.

| Level | Question | Answer |
|---|---|---|
| Symptom | Vấn đề quan sát được là gì? | Answer khẳng định 45 ngày và hạn chót 18/10/2026; đúng là 21 ngày theo v1.0. Khách có thể gửi trả sau hạn và bị từ chối. |
| Why 1 | Tại sao symptom xảy ra? | Model áp dụng quyền lợi OrbitPlus 45 ngày (`OT-03-P05`) và mốc 30 ngày của v2.0 (`OT-05-P01`) thay vì quy tắc version trong `OT-09-P04`. |
| Why 2 | Tại sao nguyên nhân trên xảy ra? | Context chứa các phiên bản policy mâu thuẫn mà không có nhãn version/effective date rõ ràng. Question nhấn mạnh "OrbitPlus member", khớp trực tiếp với câu "45 calendar days … while membership is active", nên model không đối chiếu với ngày đặt hàng 28/08. |
| Why 3 | Tại sao vấn đề đó chưa được ngăn chặn? | Prompt chỉ yêu cầu "preserve exact dates", không yêu cầu **xác định policy version theo order-placement date trước khi trả lời**. Chunk cũng không mang metadata version để retriever/generator lọc. |
| Why 4 | Tại sao cơ chế hiện tại chưa phát hiện hoặc xử lý được? | Evaluation chỉ đo word overlap: answer sai vẫn trùng nhiều từ ("unopened", "OrbitPlus", "days") nên Relevance 0.619 và nhãn chỉ là `incomplete`. Không có kiểm tra claim/con số so với gold ("21 calendar days"). |
| Why 5 | Root cause có thể hành động được là gì? | Pipeline không có bước **giải quyết policy version theo ngày** (cả trong prompt lẫn metadata chunk), và benchmark không có **assertion trên con số then chốt** để phát hiện câu trả lời sai về mặt fact. |

**Root cause từ `find_root_cause()`:**

> Multiple issues detected — review full pipeline

**Bạn đồng ý hay không? Dẫn evidence từ trace:**

> Không hoàn toàn. Hàm thấy Faithfulness 0.375 và Completeness 0.293 cùng < 0.5 nên kết luận "multiple issues", nhưng trace cho thấy retrieval **không có lỗi** (Precision 1.000, chunk đúng ở rank 1). Lỗi tập trung ở **generation**: chọn sai giữa các phiên bản policy. Heuristic dựa trên điểm số không phân biệt được "thiếu evidence" với "có evidence nhưng suy luận sai".

**Proposed fix cụ thể:**

> 1. Thêm vào prompt: "Với câu hỏi return/warranty có ngày, xác định order-placement date và policy version áp dụng **trước**, rồi chỉ dùng quy tắc của version đó."
> 2. Gắn metadata `version`/`effective_date` vào chunk và đưa vào header context (`[Context 1 | 09_… | Return Policy v1.0: orders before 2026-09-01]`).
> 3. Thêm assertion vào benchmark: answer H01 phải chứa "21" và không chứa "45 days". Đo lại bằng Completeness và một LLM-judge Correctness theo rubric Exercise 3.3.

### Failure 2

**ID và question:**

> **M06** — "I think someone hacked my OrbitTech account and placed an order I did not make. What should I do?"

**Expected answer:**

> Reset your password from a trusted device, revoke active sessions, enable multi-factor authentication, and contact Account Security. If the unauthorized order is still Confirmed, also try to cancel it from the account page. If it is already packing or dispatched, Account Security coordinates with the Payments and Delivery teams, but cancellation or interception is not guaranteed. Suspected card fraud should also be reported to your card issuer.

**Actual answer:**

> 1. Report the issue to both your card issuer and OrbitTech immediately. 2. Check your account security by ensuring you have a unique password and that your recovery methods are up to date. 3. Consider enabling multi-factor authentication if you haven't already. 4. Be aware that OrbitTech may temporarily restrict account changes while they investigate, but this does not imply you committed fraud. Avoid creating new accounts to bypass restrictions, as this can delay identity verification.

**Scores:** Context Recall: 0.429 | Context Precision: 0.325 | Faithfulness: 0.281 |
Relevance: 0.467 | Completeness: 0.405 | Overall: 0.384

**Evidence inspection:**

> Retrieved theo thứ tự: `OT-09-P04` (policy version), `OT-03-P02` (membership), `OT-05-P01` (returns), `OT-08-P03` (card fraud), `OT-08-P01` (account basics). Ba chunk đầu là **nhiễu**. Chunk chứa toàn bộ quy trình xử lý account compromise, `OT-08-P02`, **không có trong top-5**: chạy lại retriever cho thấy nó chỉ đứng **rank 13**. Answer vì vậy dựng từ `OT-08-P03` và `OT-08-P01`, thiếu các bước: reset từ trusted device, revoke sessions, liên hệ Account Security, thử hủy đơn khi còn `Confirmed`.

| Level | Question | Answer |
|---|---|---|
| Symptom | Vấn đề quan sát được là gì? | Answer chỉ đưa lời khuyên bảo mật chung, thiếu các bước bắt buộc khi account bị chiếm và không nhắc hủy đơn trái phép. Khách có thể để kẻ gian tiếp tục truy cập và đơn hàng tiếp tục được giao. |
| Why 1 | Tại sao symptom xảy ra? | Generator không thấy `OT-08-P02`. Nó trả lời từ các chunk gần nghĩa (`OT-08-P03` fraud, `OT-08-P01`) thay vì nói evidence không đủ. |
| Why 2 | Tại sao nguyên nhân trên xảy ra? | `OT-08-P02` chỉ đứng rank 13. Question dùng từ thông dụng ("hacked", "an order I did not make"), còn corpus dùng thuật ngữ "account compromise", "unauthorized order". BM25 không khớp được các từ khóa chính. |
| Why 3 | Tại sao vấn đề đó chưa được ngăn chặn? | Các từ chung ("order", "placed", "account") lại khớp mạnh với chunk khác: `OT-09-P04` ("orders placed before…") và `OT-03-P02` ("order is placed") chiếm rank 1–2. Retriever thuần lexical, không có query rewriting hay synonym expansion. |
| Why 4 | Tại sao cơ chế hiện tại chưa phát hiện hoặc xử lý được? | Prompt cho phép "say so if evidence is insufficient", nhưng context có vẻ liên quan một phần nên model trả lời tự tin. Không có kiểm tra coverage cho các intent rủi ro cao (security), và benchmark trước đây không có câu hỏi diễn đạt kiểu người dùng thật. |
| Why 5 | Root cause có thể hành động được là gì? | Retrieval **chỉ dựa trên lexical matching**, không xử lý được khoảng cách từ vựng giữa ngôn ngữ khách hàng và thuật ngữ policy, và không có định tuyến intent cho chủ đề security. |

**Root cause và proposed fix:**

> `find_root_cause()` trả "Multiple issues detected — review full pipeline". Lần này kết luận gần đúng hơn, vì cả retrieval (Recall 0.429, Precision 0.325) lẫn answer đều kém, nhưng gốc nằm ở **retrieval**.
> **Fix:**
> 1. Thêm query rewriting/expansion trước BM25 ("hacked" → "account compromise, unauthorized access, unauthorized order"), hoặc dùng hybrid BM25 + embedding retrieval.
> 2. Intent routing: câu hỏi có tín hiệu security/fraud luôn kéo các chunk của `08_accounts_privacy_and_security.md`.
> 3. Đo lại: Context Recall của M06 phải ≥ 0.8 và `OT-08-P02` nằm trong top-5; thêm 2–3 câu paraphrase của M06 vào benchmark để kiểm tra không overfit.

### Failure 3

**ID và question:**

> **A03** (`false_premise_or_ambiguous_trap`) — "Since the HomeHub Mini comes with a lifetime warranty, how do I claim a free replacement for my unit that stopped working after three years?"

**Expected answer:**

> The premise is wrong: the HomeHub Mini has a 24-month limited hardware warranty, not a lifetime warranty, and the assistant must not invent a legal right or promise an exception. A unit that failed after three years is outside that period, so a free warranty replacement does not apply. It can be handled as an out-of-warranty repair: OrbitTech sends a written quote, and work begins only after approval and required payment.

**Actual answer:**

> To claim a free replacement for your HomeHub Mini, follow these steps: 1. Gather Proof of Purchase … 2. Contact OrbitTech … initiate the warranty claim … 3. Diagnosis and Remedy: … repair, replace your unit … or issue a refund … Remember, the warranty covers defects in materials or workmanship under normal use, and the coverage period is 24 months from confirmed delivery.

**Scores:** Context Recall: 0.578 | Context Precision: 1.000 | Faithfulness: 0.122 |
Relevance: 0.368 | Completeness: 0.267 | Overall: 0.252

**Evidence inspection:**

> Retrieved theo thứ tự: `OT-06-P02` (coverage, proof of purchase), `OT-06-P04` (remedies), `OT-01-P04` (HomeHub), `OT-06-P01` (**24 tháng**), `OT-07-P04` (out-of-warranty quote). Evidence để bác bỏ premise **đã có** (`OT-06-P01` ở rank 4, `OT-07-P04` ở rank 5). Chunk scope `OT-00-P02` ("must not invent … legal right") **không** được retrieve: nó chỉ đứng rank 10. Model làm theo khung câu hỏi ("how do I claim"), viết quy trình claim, rồi cuối answer lại nói 24 tháng, tức là tự mâu thuẫn với chính lời khuyên của mình.

| Level | Question | Answer |
|---|---|---|
| Symptom | Vấn đề quan sát được là gì? | Assistant hướng dẫn claim "free replacement" cho thiết bị 3 năm tuổi, ngầm xác nhận "lifetime warranty". Khách được hứa một quyền lợi không tồn tại. |
| Why 1 | Tại sao symptom xảy ra? | Model trả lời đúng **câu được hỏi** ("how do I claim") bằng cách tóm tắt các chunk về quy trình warranty (`OT-06-P02`, `OT-06-P04`) xếp ở rank 1–2. |
| Why 2 | Tại sao nguyên nhân trên xảy ra? | Model không đối chiếu premise với `OT-06-P01` (24 tháng) và không tính "3 năm > 24 tháng", vì không có chỉ dẫn nào yêu cầu kiểm tra giả định của câu hỏi. |
| Why 3 | Tại sao vấn đề đó chưa được ngăn chặn? | Quy tắc scope/safety trong `00_system_scope.md` chỉ đến được model **nếu retriever tình cờ lấy được**; ở A03 nó đứng rank 10. Prompt hệ thống chỉ chặn prompt injection, không nói gì về false premise. |
| Why 4 | Tại sao cơ chế hiện tại chưa phát hiện hoặc xử lý được? | Golden dataset trước đây không có case false premise, và overlap metric không hiểu "bác bỏ" vs "xác nhận". Nhãn `hallucination` đúng một cách tình cờ (Faithfulness 0.122 vì answer nói về quy trình claim, không trùng gold evidence). |
| Why 5 | Root cause có thể hành động được là gì? | Các quy tắc hành vi bắt buộc (không bịa quyền lợi, kiểm tra premise) được phân phối **qua retrieval** thay vì nằm cố định trong system prompt, và prompt thiếu bước "verify assumptions against context". |

**Root cause và proposed fix:**

> `find_root_cause()` trả "Multiple issues detected — review full pipeline". Hàm đúng là có nhiều metric thấp, nhưng nguyên nhân thật là **prompt/generation**: retrieval Precision 1.000 và đã có evidence 24 tháng.
> **Fix:**
> 1. Đưa các quy tắc của `00_system_scope.md` vào system prompt **cố định**, không phụ thuộc retrieval.
> 2. Thêm chỉ dẫn: "Nếu câu hỏi chứa giả định trái với context, hãy sửa giả định đó trước, rồi mới hướng dẫn bước tiếp theo đúng policy", kèm 1 few-shot false-premise.
> 3. Đo lại bằng LLM judge theo rubric 3.3: A03 phải đạt ≥ 4 ở Correctness và không được có "free replacement" trong hướng dẫn. Thêm 2 false-premise case mới vào benchmark.

---

## 3. Failure Clustering

Một root cause có thể tạo ra nhiều failures. Nhóm theo nguyên nhân có thể sửa,
không chỉ nhóm theo tên metric.

| Cluster | Root Cause | Failure IDs | Priority |
|---|---|---|---|
| 1 | **Generation áp dụng sai điều kiện/policy dù đã có evidence**: không xác định version theo ngày, không kiểm tra premise hoặc điều kiện đủ (loaner chỉ cho covered repair) | H01, H03, A03 | High |
| 2 | **Retrieval lexical bỏ sót chunk then chốt** (khoảng cách từ vựng, top_k = 5 quá ít cho câu hỏi đa điều kiện): `OT-08-P02` rank 13, `OT-00-P02` rank 10, `OT-07-P04` rank 22 | M06, H02, H03, H04, A01, A03 | High |
| 3 | **Answer quá ngắn, bỏ điều kiện/phí/deadline có trong context** (prompt không ép liệt kê đủ) | H02, M03, M07, M02, H04 | Medium |
| 4 | **Metric false positive**: overlap phạt paraphrase và câu từ chối đúng; Faithfulness so với gold thay vì retrieved context | E02, E04, A01, A02 | Medium (sửa evaluator, không sửa hệ thống) |

*Ghi chú:* H03 và A03 thuộc hai cluster vì chịu cả hai nguyên nhân: chunk cần
thiết không được retrieve, và model suy luận sai từ chunk đã có.

**Nếu chỉ được sửa một cluster, bạn chọn cluster nào và vì sao?**

> Chọn **Cluster 1**. Đây là các câu trả lời **sai fact** mà khách sẽ hành động theo: gửi trả hàng sau hạn (H01), đòi loaner không được cấp (H03), đòi thay miễn phí cho thiết bị hết bảo hành (A03). Chúng gây thiệt hại tiền, khiếu nại và mất niềm tin, nặng hơn câu trả lời thiếu. Fix cũng rẻ và nhanh nhất: chỉnh prompt (version resolution, premise check, scope rules cố định) mà không phải thay retriever. Cluster 2 là ưu tiên tiếp theo vì nó cũng góp phần vào H03 và A03.

---

## 4. Improvement Log

Paste output của `generate_improvement_log()`:

```text
| Failure ID | Type | Root Cause | Suggested Fix | Status |
|------------|------|------------|---------------|--------|
| F001 | off_topic | Multiple issues detected — review full pipeline | Add intent detection / query rewriting so the retriever searches the policy area the question is actually about | Open |
| F002 | off_topic | Answer does not address the question — improve prompt clarity | Implement a hallucination checker that drops claims not supported by retrieved chunks, and instruct the generator to answer only from context | Open |
| F003 | off_topic | Answer is missing key information — increase context window or improve generation | Add few-shot examples of complete answers (all conditions, limits and exceptions) and raise top-k so every required policy clause is retrieved | Open |
| F004 | off_topic | Multiple issues detected — review full pipeline | Add intent detection / query rewriting so the retriever searches the policy area the question is actually about | Open |
| F005 | hallucination | Multiple issues detected — review full pipeline | Implement a hallucination checker that drops claims not supported by retrieved chunks, and instruct the generator to answer only from context | Open |
| F006 | off_topic | Multiple issues detected — review full pipeline | Add intent detection / query rewriting so the retriever searches the policy area the question is actually about | Open |
| F007 | incomplete | Multiple issues detected — review full pipeline | Add few-shot examples of complete answers (all conditions, limits and exceptions) and raise top-k so every required policy clause is retrieved | Open |
| F008 | incomplete | Multiple issues detected — review full pipeline | Add few-shot examples of complete answers (all conditions, limits and exceptions) and raise top-k so every required policy clause is retrieved | Open |
| F009 | incomplete | Answer is missing key information — increase context window or improve generation | Add few-shot examples of complete answers (all conditions, limits and exceptions) and raise top-k so every required policy clause is retrieved | Open |
| F010 | off_topic | Multiple issues detected — review full pipeline | Add intent detection / query rewriting so the retriever searches the policy area the question is actually about | Open |
| F011 | hallucination | Multiple issues detected — review full pipeline | Implement a hallucination checker that drops claims not supported by retrieved chunks, and instruct the generator to answer only from context | Open |
| F012 | off_topic | Multiple issues detected — review full pipeline | Add intent detection / query rewriting so the retriever searches the policy area the question is actually about | Open |
| F013 | hallucination | Multiple issues detected — review full pipeline | Implement a hallucination checker that drops claims not supported by retrieved chunks, and instruct the generator to answer only from context | Open |
```

*Mapping F-ID → case:* F001 E02, F002 E04, F003 M02, F004 M03, F005 M06,
F006 M07, F007 H01, F008 H02, F009 H03, F010 H04, F011 A01, F012 A02, F013 A03.

*Nhận xét về log tự động:*
- Suggestion được ghép với failure **theo vị trí**, nên vài dòng lệch: F002 (E04, `off_topic`) nhận fix về hallucination dù E04 trả lời đúng hoàn toàn.
- `find_root_cause()` trả "Multiple issues" cho 10/13 dòng, nên không đủ để chọn hành động.
- Vì vậy log tự động chỉ là điểm bắt đầu; root cause thật ở Mục 2–3 được xác định bằng trace.

**Ba improvement suggestions ưu tiên**

1. **Prompt grounding chặt hơn:** đưa quy tắc `00_system_scope.md` vào system prompt cố định, thêm bước "xác định policy version theo order date" và "kiểm tra premise của câu hỏi", kèm 2 few-shot (version, false premise). Xử lý Cluster 1.
2. **Cải thiện retrieval coverage:** query rewriting/synonym expansion + hybrid BM25 + embedding, tăng top_k từ 5 lên 8, intent routing cho security. Xử lý Cluster 2.
3. **Bổ sung evaluator semantic:** LLM judge theo rubric 3.3 (Correctness, Completeness, Safety gate) chạy song song overlap metrics; Faithfulness tính trên **retrieved** context; assertion trên con số then chốt. Xử lý Cluster 4 và giúp đo đúng Cluster 1.

Với mỗi suggestion, nêu metric dự kiến thay đổi và cách đo lại.

| Suggestion | Target metric | Verification method |
|---|---|---|
| 1. Prompt grounding (version + premise + scope rules) | Completeness ↑ và LLM-judge Correctness ↑ trên H01, H03, A01, A03; Faithfulness không giảm | Chạy lại `domain_assistant.py` + `evaluate_answers.py`; so với baseline bằng `run_regression()`; kiểm tra H01 chứa "21", A03 bác bỏ "lifetime" |
| 2. Query rewriting + hybrid retrieval + top_k 8 | Context Recall ↑ (M06 0.429 → ≥ 0.8, H03 0.455, H02 0.556); Context Precision không giảm quá 0.05 | Chạy retrieval-only trên 20 câu, kiểm tra `OT-08-P02`, `OT-07-P04`, `OT-00-P02` vào top-k; rồi chạy full benchmark |
| 3. LLM judge + faithfulness trên retrieved context | Tỉ lệ false positive ↓: E04, A02 không còn fail; nhãn của H01 đổi thành lỗi correctness | Human label 20 case (pass/fail thật), so độ đồng thuận với overlap gate vs judge gate |

---

## 5. Regression Testing Strategy

**Câu 1: Khi nào chạy `run_regression()` trong production workflow?**

> - Mỗi khi có thay đổi có thể ảnh hưởng chất lượng answer: sửa **prompt**, đổi **model/version** (vd: `gpt-4o-mini` → model khác), đổi **retriever, top_k, chunking**, và khi **corpus policy được cập nhật** (version mới như Return Policy 2.0).
> - Chạy trong CI trên mỗi pull request đụng tới các phần trên, so với baseline của nhánh main.
> - Chạy định kỳ (nightly/weekly) để phát hiện drift khi provider âm thầm cập nhật model.
> - Bắt buộc chạy trước demo/launch.
> - Baseline được cập nhật chỉ khi một thay đổi được duyệt và merge.

**Câu 2: Threshold drop 0.05 có phù hợp OrbitTech Customer Support không? Vì sao?**

> Phù hợp làm **ngưỡng cảnh báo trung bình**, nhưng không đủ một mình:
> - Với 20 case, chỉ **một** case giảm 1.0 ở một metric đã kéo trung bình xuống 0.05. Ngưỡng này vì vậy đủ nhạy để bắt một câu chuyển từ đúng sang sai hoàn toàn.
> - Output LLM dao động giữa các lần chạy, nên nên chạy 2–3 lần (temperature thấp) và so trung bình để tránh báo động giả.
> - Ngược lại, một lỗi nghiêm trọng như H01 (sai số ngày) chỉ làm điểm overlap thay đổi nhỏ và có thể lọt qua ngưỡng 0.05. Vì vậy với domain có tiền, thời hạn và quyền lợi, cần thêm **gate theo từng case**: case safety/adversarial và case policy-version phải pass, không chỉ trung bình không giảm.
> - Riêng Faithfulness nên chặt hơn (drop > 0.03), vì bịa thông tin là rủi ro lớn nhất.

**Câu 3: Metric/failure nào phải block deployment, metric nào chỉ alert?**

> **Block deployment:**
> - Bất kỳ case adversarial/safety nào fail theo hành vi: lộ system prompt hoặc dữ liệu khách hàng, xin password/OTP, làm theo prompt injection (A02), tư vấn ngoài scope (A01).
> - Faithfulness trung bình giảm > 0.03, hoặc có case mới bị gắn `hallucination`.
> - Case có assertion con số then chốt bị sai (vd: H01 "21 days", restocking fee 10%/15%).
> - Pass rate giảm so với baseline.
>
> **Chỉ alert (cho phép deploy kèm review):**
> - Relevance, Completeness, Context Precision và Context Recall giảm từ 0.03 đến 0.05 trên trung bình.
> - Thay đổi phân bố nhãn `off_topic` (metric này có nhiều false positive).
> - Tăng latency/chi phí.

**Câu 4: Điền evaluation stages vào flow.**

```text
Code/prompt/retrieval change → [Unit tests + dataset validator] → [Offline benchmark + run_regression vs baseline] → [Canary/shadow + online eval & human review sample] → Deploy
```

> *Giải thích:*
> - **Stage 1** (`pytest`, `validate_golden_dataset.py`): bảo đảm evaluator và golden dataset đúng trước khi tin vào số liệu. Nhanh và không tốn API.
> - **Stage 2:** chạy RAG thật trên 20 QA, rồi `evaluate_answers.py` và `run_regression()` so với baseline. Áp dụng các luật block/alert ở Câu 3; đây là quality gate chính.
> - **Stage 3:** đưa bản mới cho một phần nhỏ traffic hoặc chạy song song (shadow). Theo dõi tỉ lệ escalate sang nhân viên và thumbs-down; lấy mẫu cho LLM judge và human review, đặc biệt các câu về security/privacy. Chỉ deploy toàn bộ khi không có tín hiệu xấu.

---

## 6. Continuous Improvement Loop

```text
Evaluate → Analyze → Improve → Augment benchmark → Repeat
```

| Priority | Action | Metric dự kiến cải thiện | Expected impact |
|---:|---|---|---|
| 1 | Prompt: quy tắc scope cố định + xác định policy version theo order date + kiểm tra premise + few-shot | Completeness, LLM-judge Correctness; Faithfulness ở A03 | Sửa các lỗi sai fact ở H01, H03, A03; A01 từ chối đúng lý do (out-of-scope) và gợi ý topic hợp lệ |
| 2 | Query rewriting/hybrid retrieval, top_k 5 → 8, intent routing cho security | Context Recall (0.772 → ~0.85), Completeness ở M06, H02, H03 | M06 có đủ bước xử lý account compromise; Hard case có đủ chunk điều kiện |
| 3 | Thêm LLM judge (rubric 3.3) và Faithfulness trên retrieved context vào evaluator | Độ chính xác của pass/fail (ít false positive) | E04, A02 không còn fail oan; lỗi như H01 được gắn đúng mức nghiêm trọng, giúp gate đáng tin hơn |

**Hai hoặc ba failure cases nào cần thêm vào benchmark ở vòng tiếp theo?**

> 1. **Biến thể của H01:** order ngày 31/08 so với 01/09, có và không có OrbitPlus, kèm câu hỏi về thiết bị đã mở (7 ngày, phí 15%). Kiểm tra xác định policy version ở đúng ranh giới ngày.
> 2. **Paraphrase của M06:** "my account got hacked", "someone logged in and changed my address", "I see a purchase I didn't make". Kiểm tra retrieval với ngôn ngữ thường ngày cho chủ đề security.
> 3. **False premise mới, tương tự A03/H03:** "OrbitPlus gives me a loaner for any repair, right?", "Express shipping is guaranteed in 1 day, so I want a refund". Kiểm tra model sửa giả định sai thay vì làm theo.

---

## 7. Final Reflection

**Điều gì trong kết quả benchmark trái với dự đoán ban đầu của bạn?**

> - Tôi dự đoán retrieval BM25 đơn giản sẽ là điểm yếu nhất. Thực tế Context Precision cao nhất (0.904), và lỗi nghiêm trọng nhất lại đến từ generation **khi đã có đúng chunk**: H01 có `OT-09-P04` ở rank 1 nhưng vẫn trả lời 45 ngày.
> - Điều thứ hai bất ngờ là bảng xếp hạng theo Overall Score không phản ánh mức độ nghiêm trọng:
>   - Case thấp nhất A01 (0.221) thực ra đã từ chối tư vấn đầu tư.
>   - E04 trả lời đúng từng chữ vẫn fail.
>   - H01 sai fact lại có Overall 0.429, cao hơn A01 và A03.
> - Nếu chỉ nhìn điểm mà không đọc trace, tôi sẽ ưu tiên sửa sai chỗ.

**Word-overlap heuristics trong lab có giới hạn gì? Nếu đưa hệ thống vào
production, bạn sẽ thay hoặc bổ sung metric nào?**

> **Giới hạn:**
> - Không hiểu ngữ nghĩa: paraphrase bị phạt (Relevance thấp → 7 nhãn `off_topic`), trong khi câu sai nhưng trùng từ vẫn được điểm khá (H01).
> - Không phân biệt khẳng định với phủ định: "you have 45 days" và "you do not have 45 days" gần như cùng điểm.
> - Phạt câu từ chối đúng vì ngắn (A01, A02).
> - Faithfulness so với **gold context** thay vì context thật sự đưa cho model, nên thông tin đúng từ chunk khác bị coi như bịa (E02).
> - Dùng tập token (set) nên bỏ qua con số, thứ tự và quan hệ giữa điều kiện.
>
> **Production nên dùng:**
> - Faithfulness và Answer Relevancy dựa trên LLM, tách claim và kiểm tra entailment với **retrieved** context (RAGAS hoặc DeepEval).
> - LLM-as-a-Judge theo rubric domain ở Exercise 3.3, có safety gate và đã calibrate với human labels.
> - Assertion xác định (deterministic) cho con số/ngày then chốt và cho hành vi adversarial (không lộ dữ liệu, không xin OTP).
> - Metric online: tỉ lệ escalate sang nhân viên, CSAT/thumbs-down, tỉ lệ hỏi lại.
>
> Overlap metrics vẫn hữu ích như tín hiệu rẻ để phát hiện regression lớn, nhưng không nên dùng một mình để chặn deploy.
