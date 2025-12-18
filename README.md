
├── data/                                # Thư mục chứa dữ liệu (Dataset)
│  
├── notebook/                            # Thư mục chứa Jupyter Notebooks (Mã nguồn chính)
│   ├── Lab5_part3.pdf                    
│
├── README.md                           # File báo cáo chi tiết này
│
├── part3                                # Mã nguồn Python (Modules/Classes tái sử dụng)
│   
├── .gitignore                           # File cấu hình bỏ qua file rác (tmp, __pycache__)

Implementation (50%) – viết code hoàn thiện các task.

Report & Analysis (50%) – mô tả các bước thực hiện, phân tích kết quả, so sánh performance.

Part 1: Implementation (50%)
  Task 1: Tải và Tiền xử lý Dữ liệu

      Đọc dữ liệu từ file .conllu (train/dev) bằng hàm load_conllu().
      
      Mỗi câu được biểu diễn dưới dạng danh sách các cặp (word, UPOS_tag).
      
      Xây dựng từ điển word_to_ix và tag_to_ix từ tập train, thêm token <UNK> cho từ chưa biết.
      
      Kích thước từ vựng: 19,675 từ
      
      Số nhãn UPOS: 17 nhãn

  Task 2: Tạo PyTorch Dataset và DataLoader

      Tạo lớp POSDataset kế thừa từ torch.utils.data.Dataset để trả về (sentence_indices, tag_indices).
      
      Viết hàm collate_fn để padding các câu về cùng độ dài batch sử dụng torch.nn.utils.rnn.pad_sequence.
      
      Khởi tạo DataLoader:
      
      Train batches: 392
      
      Dev batches: 63

  Task 3: Xây dựng Mô hình RNN

      Xây dựng mô hình SimpleRNNForTokenClassification gồm:
      
      nn.Embedding – chuyển chỉ số từ sang vector embedding 128 chiều.
      
      nn.RNN – xử lý chuỗi vector embedding với hidden_size=128.
      
      nn.Linear – ánh xạ output RNN sang 17 nhãn UPOS.

  Task 4: Huấn luyện Mô hình

      Sử dụng nn.CrossEntropyLoss(ignore_index=PAD_TAG) và Adam optimizer.
      
      Huấn luyện 5 epoch với batch_size=32:
      
      Loss giảm dần: 1.050 → 0.270

  Task 5: Đánh giá Mô hình
    
      Train Accuracy: 0.9305
      
      Dev Accuracy: 0.8708

Part 2: Report & Analysis (50%)


Implementation Steps

      Load dữ liệu .conllu → xử lý thành danh sách câu (word, tag).
      
      Xây dựng từ điển từ dữ liệu train.
      
      Tạo Dataset & DataLoader với padding.
      
      Xây dựng mô hình RNN cho POS tagging.
      
      Huấn luyện mô hình với CrossEntropyLoss.
      
      Đánh giá mô hình trên tập dev.
      
      Dự đoán câu mới.

Code Execution Guide

  Source code: 
  
      src/main/Lap5-3

  Cài đặt thư viện:

      pip install torch torchvision


  Tải dữ liệu UD English-EWT:

      en_ewt-ud-train.conllu
      
      en_ewt-ud-dev.conllu

  Chạy notebook hoặc script:

      python lab5_pos_rnn.py


  Quan sát output:

      Loss sau mỗi epoch
      
      Accuracy train/dev
      
      Dự đoán câu mới với predict_sentence()

Result Analysis

  Baseline (Logistic Regression):

      Accuracy train/dev ~ 0.82 / 0.78

  Improved Model (RNN):

      Accuracy train/dev: 0.9305 / 0.8708

  Analysis:

      RNN capture ngữ cảnh trong chuỗi tốt hơn Logistic Regression.
      
      Chênh lệch train/dev ~6% → có thể do overfitting nhẹ.

  Challenges & Solutions

      Padding các câu khác độ dài → sử dụng pad_sequence và ignore_index trong loss.
      
      Từ hiếm không có trong vocab → thêm <UNK>.
      
      Batch lớn/seq dài → chọn batch_size=32 để quản lý memory.
  
  References
      
      PyTorch RNN Documentation
      
      Universal Dependencies English-EWT
      
      PyTorch CrossEntropyLoss ignore_index
