README - Thesis Data 說明
=======================

本資料夾整理論文使用的實驗資料、程式碼與結果統計。


一、01_Data
-----------

放各部實驗影片的資料。

每部影片資料夾主要包含：

- 原始影片
- frames：逐影格圖片
- annotations.csv：Ground Truth 標註
- result4：當時執行程式產生的實驗結果


二、02_Code
-----------

放目前整理後的主要程式。

run_pipeline.py
主程式

stage1.py
Stage 1 高速運動初始軌跡追蹤。

stage2.py
Stage 2 微小目標追蹤，包含 LBP、LTP、CSWC。

stage3.py
Stage 3 軌跡復原。

stage3_offline_eval.py
Stage 3 離線曲線實驗與結果統計。

程式內部分檔案或方法名稱使用 weighted，對應論文中的 CSWC。


三、03_Summary
--------------

raw
保存原始實驗結果的彙整資料

paper_result
整理論文中使用的主要實驗結果，包含：

- Stage 1 結果
- Stage 1 與 Stage 2 交接結果
- Stage 2 三種方法比較
- Stage 3 offline 實驗
- 各影片結果與詳細檢查資料


四、程式使用方式
----------------

一般追蹤流程使用：

run_pipeline.py


Stage 3 offline 實驗使用：

stage3_offline_eval.py

此程式會讀取已產生的 Stage 2 結果，再進行下落軌跡曲線擬合與統計，
不需要重新執行影像追蹤。


五、結果與版本說明
------------------

03_Summary/raw
保存原始實驗結果。

03_Summary/paper_result
保存整理後、對應論文使用的結果。

