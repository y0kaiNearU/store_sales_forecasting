# Walmart Recruiting - Store Sales Forecasting

Kaggle კონკურსი: [Walmart Recruiting - Store Sales Forecasting](https://www.kaggle.com/competitions/walmart-recruiting-store-sales-forecasting)

ეს არის ორკაციანი გუნდის საერთო რეპოზიტორია. დავალება ორ ნაწილად იყოფა არქიტექტურის ტიპის მიხედვით -cთითოეული ნაწილი ცალკე ავტორისაა და ქვემოთ ცალკე სექციადაა აღწერილი. ორივე ნაწილი ერთსა და იმავე W&B პროექტში (`store_sales_forecast`) იტვირთება.

---
## ამოცანის აღწერა

ამოცანის მიზანია Walmart-ის მაღაზიებისა და დეპარტამენტებისთვის მომავალი კვირების გაყიდვების პროგნოზირება.

პროგნოზი კეთდება თითოეული Store + Dept + Date კომბინაციისთვის. Target ცვლადია Weekly_Sales.

მონაცემები შედგება შემდეგი ფაილებისგან:

- train.csv — ისტორიული გაყიდვები
- test.csv — მომავალი თარიღები, რომლებზეც უნდა გაკეთდეს პროგნოზი
- features.csv — დამატებითი ცვლადები, როგორიცაა temperature, fuel price, markdowns, CPI, unemployment და holiday flag
- stores.csv — მაღაზიის ტიპი და ზომა
- sampleSubmission.csv — Kaggle submission-ის ფორმატი

შეფასების მეტრიკაა Weighted Mean Absolute Error, ანუ WMAE.

Holiday კვირებზე შეცდომა უფრო მძიმეა, რადგან ამ კვირების წონა არის 5, ხოლო ჩვეულებრივი კვირების წონა არის 1. ამიტომ მოდელმა განსაკუთრებით კარგად უნდა იმუშაოს holiday პერიოდებზე.

---

## რეპოზიტორიის სტრუქტურა

პროექტში ძირითადი ფაილებია:

- model_experiment_ARIMA_SARIMA.ipynb
- model_experiment_DLinear.ipynb
- model_experiment_LightGBM.ipynb
- model_experiment_NBEATS.ipynb
- model_experiment_PatchTST.ipynb
- model_experiment_Prophet.ipynb
- model_experiment_TFT.ipynb
- model_experiment_TimesFM.ipynb
- model_experiment_XGBoost.ipynb
- model_inference.ipynb
- src/config.py
- src/data_loading.py
- src/features.py
- src/metrics.py
- src/models.py
- src/classical.py
- src/validation.py
- src/wandb_utils.py
- src/preprocessing.py
- requirements.txt
- pyproject.toml
- README.md

data/raw ფოლდერში ლოკალურად უნდა მოთავსდეს Kaggle-ის ფაილები:

- train.csv
- test.csv
- features.csv
- stores.csv
- sampleSubmission.csv

ეს მონაცემები GitHub-ზე არ არის ატვირთული, რადგან Kaggle-ის raw data-ს რეპოზიტორიაში დამატება არ არის რეკომენდებული.

---

## გაშვების ინსტრუქცია

პირველ რიგში საჭიროა dependencies-ის დაყენება:

pip install -r requirements.txt

შემდეგ საჭიროა W&B login:

wandb login

სანამ notebook-ებს გავუშვებთ, შესაძლებელია smoke test-ის გაშვება:

python run_smoke_test.py

ყველაზე მნიშვნელოვანი notebook-ებია LightGBM, XGBoost და model_inference, რადგან საბოლოო submission tree-based საუკეთესო მოდელით გენერირდება.

_(`pyproject.toml`/`uv.lock` არსებობს რეპოზიტორიის root-ში და ორივე ნაწილის დამოკიდებულებებს მოიცავს — `uv sync` ალტერნატივაცაა `pip install -r requirements.txt`-ის ნაცვლად.)_

---

# ნაწილი 1: Tree-Based & კლასიკური მოდელები


ეს ნაწილი მოიცავს ორ ძირითად მიმართულებას:

1. Tree-Based Models: LightGBM, XGBoost
2. Classical Statistical Time-Series Models: Seasonal Naive baseline, SARIMA, Prophet

ექსპერიმენტები დალოგილია Weights & Biases-ზე.

W&B entity: gchal22-free-university-of-tbilisi-  
W&B project: store_sales_forecast  
W&B project URL: https://wandb.ai/gchal22-free-university-of-tbilisi-/store_sales_forecast

MLflow ამ ვერსიაში არაა გამოყენებული. Model Registry-ის მოთხოვნა შესრულებულია W&B Artifacts-ის საშუალებით. საუკეთესო მოდელი ინახება W&B artifact-ად, ხოლო model_inference.ipynb პირდაპირ იქიდან ტვირთავს მოდელს და აგენერირებს Kaggle submission-ს.

---

## Validation მიდგომა

ამ ამოცანაში random split არასწორია, რადგან მონაცემები time-series ტიპისაა. თუ მომავალ თარიღებს random split-ით შევურევთ train და validation ნაწილებში, მივიღებთ data leakage-ს და არარეალურად კარგ შედეგს.

ამის ნაცვლად გამოვიყენეთ chronological expanding-window validation.

იდეა ასეთია:

- Fold 1: ძველი თარიღებით ვატრენინგებთ მოდელს და შემდეგ 8 კვირაზე ვამოწმებთ
- Fold 2: train პერიოდი იზრდება და შემდეგი 8 კვირა გამოიყენება validation-ად
- Fold 3: კიდევ უფრო დიდი train პერიოდი გამოიყენება და მომდევნო 8 კვირა მოწმდება

ეს მიდგომა უფრო ახლოს არის რეალურ Kaggle სიტუაციასთან, სადაც წარსული მონაცემებით მომავალი კვირები უნდა ვიწინასწარმეტყველოთ.

---

## Feature Engineering

Tree-based მოდელებისთვის time-series ამოცანა გარდაიქმნა supervised regression ამოცანად. ამისთვის თითოეულ Store + Dept + Date ჩანაწერს დაემატა ისტორიული, კალენდარული და სტატისტიკური feature-ები.

გამოყენებული ძირითადი feature-ებია:

- Store
- Dept
- IsHoliday
- TypeOrdinal
- Size
- Temperature
- Fuel_Price
- MarkDown1
- MarkDown2
- MarkDown3
- MarkDown4
- MarkDown5
- CPI
- Unemployment

კალენდარული feature-ები:

- year
- month
- week
- day
- quarter
- dayofyear
- is_month_start
- is_month_end
- week_sin
- week_cos
- month_sin
- month_cos

Holiday და retail period feature-ები:

- is_super_bowl_window
- is_labor_day_window
- is_thanksgiving_window
- is_christmas_window
- is_black_friday_week

Lag feature-ები:

- lag_1
- lag_2
- lag_3
- lag_4
- lag_8
- lag_13
- lag_26
- lag_52

Rolling feature-ები:

- rolling_mean_4
- rolling_mean_8
- rolling_mean_13
- rolling_mean_26
- rolling_std_4
- rolling_std_8
- rolling_std_13
- rolling_std_26

Aggregate feature-ები:

- store_dept_mean
- store_dept_median
- store_mean
- store_median
- dept_mean
- dept_median
- global_mean
- global_median

Lag და rolling feature-ები მოდელს ეხმარება წინა გაყიდვების pattern-ების დაჭერაში. Aggregate feature-ები კი აძლევს ზოგად ინფორმაციას კონკრეტული მაღაზიის, დეპარტამენტის და მათი კომბინაციის საშუალო გაყიდვებზე.

---

## Pipeline მიდგომა

დავალების მოთხოვნის მიხედვით, საუკეთესო მოდელი უნდა იყოს შენახული pipeline-ად და უნდა ეშვებოდეს პირდაპირ raw test set-ზე.

ამისთვის src/models.py-ში გამოყენებულია WalmartSalesForecaster class.

ეს object ინახავს:

- trained model-ს
- train history-ს lag და rolling feature-ებისთვის
- features.csv და stores.csv metadata-ს
- feature column list-ს
- missing value fallback-ებს
- aggregate statistics-ს

ამიტომ inference-ის დროს raw test.csv-ს ხელით preprocessing არ სჭირდება. model_inference.ipynb ტვირთავს მოდელს W&B artifact-დან, კითხულობს raw test.csv-ს და აგენერირებს submission_best_model.csv ფაილს.

---

## მოდელები

## Seasonal Naive baseline

Seasonal Naive არის მარტივი baseline მოდელი. ის პროგნოზად იყენებს იმავე Store + Dept კომბინაციის გაყიდვებს 52 კვირის წინ.

მისი უპირატესობაა სიმარტივე და სისწრაფე. Weekly sales ამოცანაში ეს ძლიერი baseline-ია, რადგან retail გაყიდვებში წლიური seasonality ხშირად მნიშვნელოვანია.

ნაკლოვანებები:

- ვერ იყენებს markdowns, CPI, unemployment, store size და სხვა external feature-ებს
- ცუდად რეაგირებს ცვლილებებზე
- არასრული history-ის მქონე სერიებისთვის fallback სჭირდება

W&B run: SeasonalNaive_CV

---

## SARIMA

SARIMA არის classical statistical time-series მოდელი, რომელიც იყენებს autoregressive, differencing, moving average და seasonal კომპონენტებს.

Walmart dataset-ზე SARIMA-ს მთავარი პრობლემა არის scale. გვაქვს ბევრი Store + Dept time-series, და თითოეულისთვის ცალკე SARIMA მოდელის training ძალიან ნელია.

ამიტომ SARIMA გამოვიყენე მხოლოდ representative selected series-ზე, როგორც classical model comparison.

W&B run: SARIMA_Selected_Series

---

## Prophet

Prophet იყენებს trend, seasonality და holiday decomposition მიდგომას.

მისი უპირატესობაა ის, რომ მარტივია ცალკეული time-series forecasting-ისთვის და კარგად ხსნის trend/seasonality კომპონენტებს.

Walmart-ის ამოცანაში Prophet ნაკლებად პრაქტიკულია, რადგან გვაქვს ბევრი Store + Dept სერია. ყველა მათგანზე ცალ-ცალკე მოდელის დატრენინგება რთული და ნელია. გარდა ამისა, Prophet ვერ იყენებს cross-series tabular feature-ებს ისე ეფექტურად, როგორც LightGBM ან XGBoost.

W&B run: Prophet_Selected_Series

---

## LightGBM

LightGBM არის gradient boosting decision tree მოდელი. ამ ამოცანისთვის ის კარგიი არჩევანია, რადგან Walmart-ის data ბევრი tabular feature-ისგან შედგება.

LightGBM-ის უპირატესობები:

- კარგად მუშაობს დიდ tabular dataset-ზე
- ეფექტურად იყენებს lag და rolling feature-ებს
- შეუძლია non-linear relationship-ების დაჭერა
- სწრაფია XGBoost-თან შედარებით
- sample weights-ის საშუალებით შესაძლებელია holiday კვირებისთვის მეტი მნიშვნელობის მინიჭება

LightGBM-ის validation შედეგი:

Validation WMAE: 1474.88338

W&B runs:

- LightGBM_CV
- LightGBM_Final_Model
- LightGBM_Submission

Artifacts:

- lightgbm-pipeline:latest
- best-model:latest
- lightgbm-submission:latest

---

## XGBoost

XGBoost ასევე gradient boosting tree model-ია. ის ძალიან ძლიერი მოდელია tabular data-ზე, თუმცა ამ ექსპერიმენტში LightGBM-ზე ოდნავ სუსტი შედეგი აჩვენა.

XGBoost-ის validation შედეგი:

Validation WMAE: 1493.66349

W&B runs:

- XGBoost_CV
- XGBoost_Final_Model
- XGBoost_Submission

Artifacts:

- xgboost-pipeline:latest
- xgboost-submission:latest

---

## W&B logging structure

ექსპერიმენტები W&B-ზე დალოგილია model group-ების მიხედვით.

LightGBM group:

- LightGBM_CV
- LightGBM_Final_Model
- LightGBM_Submission

XGBoost group:

- XGBoost_CV
- XGBoost_Final_Model
- XGBoost_Submission

Classical group:

- SeasonalNaive_CV
- SARIMA_Selected_Series
- Prophet_Selected_Series

Inference run:

- Best_Model_Inference

W&B-ზე ლოგირდება:

- fold-level WMAE
- CV mean WMAE
- CV standard deviation
- model hyperparameters
- prediction summary statistics
- validation result CSV artifacts
- model joblib artifacts
- submission CSV artifacts

---

## დასკვნა

ამ ამოცანაზე classical models სასარგებლოა baseline-ისა და თეორიული შედარებისთვის, მაგრამ full Walmart forecasting-ზე tree-based models უფრო პრაქტიკულია.

მთავარი მიზეზებია:

1. Dataset შედგება ბევრი parallel time series-ისგან.
2. პროგნოზი დამოკიდებულია არა მხოლოდ წინა გაყიდვებზე, არამედ external features-ზეც.
3. Holiday weeks მეტრიკაში უფრო მნიშვნელოვანია, რაც sample weighting-ით tree-based მოდელებში მარტივად გავითვალისწინეთ.
4. LightGBM და XGBoost კარგად იყენებენ lag, rolling და aggregate feature-ებს.
5. Classical models თითოეულ სერიაზე ცალკე fit-ს საჭიროებს, რაც ნელია და inference-ს ართულებს.

საბოლოოდ, ჩემს მიერ გატესტილ მოდელებში საუკეთესო შედეგი მიიღო LightGBM-მა. მან XGBoost-ზე უკეთესი validation WMAE აჩვენა და სწორედ LightGBM pipeline გამოვიყენე Kaggle submission-ისთვის.

---

# ნაწილი 2: Deep Learning მოდელები

ეს ნაწილი მოიცავს `neuralforecast`-ზე დაფუძნებულ ოთხ არქიტექტურას (N-BEATS, DLinear, PatchTST, TFT) და ბონუს foundation model-ს (TimesFM). ექსპერიმენტები ასევე W&B-ზეა დალოგილი (იგივე პროექტი, `store_sales_forecast`), ცალკე group-ებით.

## deep learning-ის ნაწილის ფაილები

- `model_experiment_NBEATS.ipynb`, `model_experiment_DLinear.ipynb`, `model_experiment_PatchTST.ipynb`, `model_experiment_TFT.ipynb`, `model_experiment_TimesFM.ipynb`
- `src/preprocessing.py` — საერთო load/feature/reshape ფუნქციები DL notebook-ებისთვის (`add_unique_id`, `to_long_format`, `build_features`, `wmae`)
- `src/dl_models.py` — pipeline wrapper კლასები (`NeuralForecastPipeline`, `TFTForecastPipeline`, `TimesFMForecastPipeline`)

## არქიტექტურული მიმოხილვა

| მოდელი | ტიპი | Exogenous ცვლადები | შენიშვნა |
|---|---|---|---|
| DLinear | წრფივი დეკომპოზიცია (trend + seasonal) | არა | ორი `Linear` შრე, ყველაზე მარტივი |
| N-BEATS | სტეკირებული MLP ბლოკები, doubly-residual | არა | 2.6M პარამეტრი, ყველაზე მაღალი capacity ამ ოთხს შორის |
| PatchTST | Transformer + patch-based tokenization | არა | ViT-ის იდეის ანალოგი დროით სერიებზე |
| TFT | LSTM + attention + variable-selection GRN | **დიახ** | ერთადერთი DL მოდელი, რომელიც რეალურად იყენებს `CPI`/`MarkDown*`/`IsHoliday`-ს |
| TimesFM (bonus) | Google-ის pretrained foundation model | არა | Zero-shot — არანაირი ტრენინგი ამ dataset-ზე |

ოთხივე `neuralforecast`-ზე დაფუძნებული მოდელი ერთსა და იმავე `H=39` (Kaggle-ის ჰორიზონტი) და `INPUT_SIZE=52` (ერთი წლის lookback) კონფიგურაციაზეა ვალიდირებული პირდაპირი შედარებადობისთვის, ერთი holdout window-ით (არა multi-fold CV — DL მოდელების თითოეული fold ხელახლა ტრენინგს მოითხოვს, რაც ღრმა ქსელებისთვის, განსაკუთრებით TFT-სთვის, ძალიან ძვირი გამოვიდოდა).

## შედეგები და ტიუნინგი

| მოდელი | Validation WMAE | ტიუნინგი |
|---|---:|---|
| **N-BEATS** | **1825.50** | `max_steps` empirical sweep (50–8000): პიკი ზუსტად 200 ნაბიჯზეა — მაღალი capacity-ის გამო სწრაფად overfit ხდება ორივე მიმართულებით |
| DLinear | 1894.02 | `max_steps` sweep (100–24000): plateau ~16000-ზე — დაბალი capacity-ის გამო (მხოლოდ ორი წრფივი შრე) ნელა converge-დება |
| PatchTST | 1932.15 | `max_steps` sweep (300–2000): plateau ~1200-ზე. `patch_len`/`stride` ალტერნატივებიც შემოწმდა — წვრილმა patch-ებმა (8/4) მცირედით აჯობა, მაგრამ 2x-ზე მეტი compute-ის ფასად, ამიტომ default (16/8) შენარჩუნდა |
| TFT | 2593.39 | ჯერ არ არის tuning-ის ეტაპზე — `hidden_size`/`n_head` CPU-ს სისწრაფისთვის შემცირებულია (128/4 → 32/2) |
| TimesFM (bonus) | 2507.66 | N/A — zero-shot, არანაირი გრადიენტული ტრენინგი ამ dataset-ზე |

**DLinear-სა და N-BEATS-ს შორის საინტერესო კონტრასტი**: სრულიად საპირისპირო tuning ქცევა აჩვენეს. DLinear-ს (დაბალი capacity) 16000 ნაბიჯი დასჭირდა კონვერგენციისთვის, ხოლო N-BEATS-მა (მაღალი capacity, სტეკირებული MLP ბლოკები) 200 ნაბიჯის შემდეგ სწრაფად overfit გახდა. ეს პირდაპირ ასახავს ორი არქიტექტურის capacity-ის განსხვავებას — DLinear-ს მეტი გრადიენტული ნაბიჯი სჭირდება, N-BEATS-ს კი მეტისმეტად სწრაფად ემახსოვრება training noise.


## Pipeline / Model Registry მიდგომა

DL pipeline-ები `src/dl_models.py`-შია - იგივე self-contained `predict(raw_df) -> DataFrame` კონტრაქტით, რასაც `WalmartSalesForecaster` იყენებს ნაწილი 1-ში. განსხვავება მხოლოდ შენახვის მექანიზმშია: `neuralforecast`-ის ობიექტები (torch/Lightning internals) `joblib`-ით საიმედოდ არ pickle-დება, ამიტომ `nf.save()`-ის native checkpoint-დირექტორია პირდაპირ ილოგება W&B artifact-ად (`src/wandb_utils.py`-ის `save_and_log_pipeline_artifact` helper-ით - TFT-სთვის მაგალითად `nf_model`-ის დირექტორია + `features.csv`/`stores.csv` ერთად ილოგება ერთ artifact-ში).

თითოეული DL არქიტექტურა საკუთარი, ცალკე W&B artifact-ის სახელით ილოგება (`dlinear-pipeline`, `nbeats-pipeline`, `patchtst-pipeline`, `tft-pipeline`, `timesfm-pipeline`) — `latest`/`candidate` ალიასებით. DL-ის საუკეთესო მოდელს (N-BEATS) დამატებით აქვს `champion` ალიასი.


## DL საუკეთესო მოდელი

**⚠️ N-BEATS (1825.50) არ სჯობს ნაწილი 1-ის LightGBM-ს (1474.88).** ორივე ნაწილი ცალკე W&B registry-ითაა წარმოდგენილი (`best-model` = LightGBM, `nbeats-pipeline:champion` = N-BEATS).

## Deep Learning დასკვნა

ხუთივე ტესტირებული DL არქიტექტურა ჩამორჩება LightGBM-ს ამ dataset-ზე, თუმცა თავად DL არქიტექტურებს შორის შედარება საინტერესო თეორიულ სურათს იძლევა:

1. **Capacity vs. tuning direction**: მარტივმა (DLinear) და რთულმა (N-BEATS) მოდელებმა საპირისპირო `max_steps` ქცევა აჩვენეს — დაბალი capacity მეტ ნაბიჯს საჭიროებს კონვერგენციისთვის, მაღალი capacity კი სწრაფად overfit ხდება.
2. **Exogenous ცვლადების ხელმისაწვდომობა**: DLinear/N-BEATS/PatchTST საერთოდ ვერ იყენებენ `CPI`/`MarkDown*`/`IsHoliday`-ს — მხოლოდ TFT-ს შეუძლია, რაც მას თეორიულად ყველაზე ახლოს აყენებს LightGBM-თან feature-გამოყენების კუთხით, თუმცა practice-ში ყველაზე ცუდი შედეგი აქვს ამ ხუთს შორის.
3. **Foundation model-ის კონკურენტუნარიანობა**: TimesFM-მა (ტრენინგის გარეშე) TFT-ს



---

## საბოლოო inference

საბოლოო submission კეთდება model_inference.ipynb-ში.

ეს notebook აკეთებს შემდეგს:

1. W&B-დან ტვირთავს best-model:latest artifact-ს
2. კითხულობს raw test.csv-ს
3. იძახებს model.predict(test)
4. ქმნის submission_best_model.csv ფაილს
5. submission-ს ლოგავს W&B artifact-ად

ამჟამად best-model:latest არის LightGBM, რადგან მას ჰქონდა საუკეთესო validation WMAE.

---

## შედეგები

| მოდელი | Validation WMAE | Kaggle Public Score | Kaggle Private Score | შენიშვნა |
|---|---:|---:|---:|---|
| Seasonal Naive | baseline only | - | - | გამოყენებულია baseline შედარებისთვის |
| SARIMA selected series | selected series only | - | - | ნელია ყველა Store + Dept სერიაზე |
| Prophet selected series | selected series only | - | - | representative classical experiment |
| **LightGBM** | **1474.88338** | **2954.97771** | **3131.03223** | საუკეთესო მოდელი და final submission |
| XGBoost | 1493.66349 | - | - | კარგი შედეგი, მაგრამ LightGBM-ზე ოდნავ სუსტი |
| N-BEATS | 1825.50 | - | - | საუკეთესო DL მოდელი, `max_steps` tuned |
| DLinear | 1894.02 | - | - | უმარტივესი DL არქიტექტურა, tuned |
| PatchTST | 1932.15 | - | - | tuned |
| TimesFM (bonus) | 2507.66 | - | - | zero-shot, ტრენინგის გარეშე |
| TFT | 2593.39 | - | - | ერთადერთი DL მოდელი exogenous ცვლადებით, ჯერ untuned |

---

## საბოლოო შედეგი Kaggle-ზე

Final submission გაკეთდა LightGBM მოდელით.

Kaggle Public Score: 2954.97771  
Kaggle Private Score: 3131.03223

Public და private score validation WMAE-ზე მაღალია, რაც მოსალოდნელია, რადგან Kaggle test period განსხვავდება validation split-ებისგან და მომავალ თარიღებზე forecast უფრო რთულია. მიუხედავად ამისა, LightGBM დარჩა საუკეთესო მოდელად.
