# Walmart Recruiting - Store Sales Forecasting

ეს რეპოზიტორია შექმნილია Kaggle-ის Walmart Recruiting - Store Sales Forecasting ამოცანისთვის.

ჩემი ნაწილი მოიცავს ორ ძირითად მიმართულებას:

1. Tree-Based Models: LightGBM, XGBoost
2. Classical Statistical Time-Series Models: Seasonal Naive baseline, SARIMA, Prophet

ექსპერიმენტები დალოგილია Weights & Biases-ზე.

W&B entity: gchal22-free-university-of-tbilisi-  
W&B project: store_sales_forecast  
W&B project URL: https://wandb.ai/gchal22-free-university-of-tbilisi-/store_sales_forecast

MLflow ამ ვერსიაში არ გამოვიყენე. Model Registry-ის მოთხოვნა შესრულებულია W&B Artifacts-ის საშუალებით. საუკეთესო მოდელი ინახება W&B artifact-ად, ხოლო model_inference.ipynb პირდაპირ იქიდან ტვირთავს მოდელს და აგენერირებს Kaggle submission-ს.

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

- model_experiment_LightGBM.ipynb
- model_experiment_XGBoost.ipynb
- model_experiment_ARIMA_SARIMA.ipynb
- model_experiment_Prophet.ipynb
- model_inference.ipynb
- src/config.py
- src/data_loading.py
- src/features.py
- src/metrics.py
- src/models.py
- src/classical.py
- src/validation.py
- src/wandb_utils.py
- requirements.txt
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

რეკომენდებული გაშვების რიგია:

1. model_experiment_LightGBM.ipynb
2. model_experiment_XGBoost.ipynb
3. model_experiment_ARIMA_SARIMA.ipynb
4. model_experiment_Prophet.ipynb
5. model_inference.ipynb

ყველაზე მნიშვნელოვანი notebook-ებია LightGBM, XGBoost და model_inference, რადგან საბოლოო submission tree-based საუკეთესო მოდელით გენერირდება.

---

## Validation მიდგომა

ამ ამოცანაში random split არასწორია, რადგან მონაცემები time-series ტიპისაა. თუ მომავალ თარიღებს random split-ით შევურევთ train და validation ნაწილებში, მივიღებთ data leakage-ს და არარეალურად კარგ შედეგს.

ამის ნაცვლად გამოვიყენე chronological expanding-window validation.

იდეა ასეთია:

- Fold 1: ძველი თარიღებით ვატრენინგებთ მოდელს და შემდეგ 8 კვირაზე ვამოწმებთ
- Fold 2: train პერიოდი იზრდება და შემდეგი 8 კვირა გამოიყენება validation-ად
- Fold 3: კიდევ უფრო დიდი train პერიოდი გამოიყენება და მომდევნო 8 კვირა მოწმდება

ეს მიდგომა უფრო ახლოს არის რეალურ Kaggle სიტუაციასთან, სადაც წარსული მონაცემებით მომავალი კვირები უნდა ვიწინასწარმეტყველოთ.

---

## Feature Engineering

Tree-based მოდელებისთვის time-series ამოცანა გარდავქმენი supervised regression ამოცანად. ამისთვის თითოეულ Store + Dept + Date ჩანაწერს დაემატა ისტორიული, კალენდარული და სტატისტიკური feature-ები.

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

LightGBM არის gradient boosting decision tree მოდელი. ამ ამოცანაზე ის ძლიერი არჩევანია, რადგან Walmart-ის data ბევრი tabular feature-ისგან შედგება.

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
| LightGBM | 1474.88338 | 2954.97771 | 3131.03223 | საუკეთესო მოდელი და final submission |
| XGBoost | 1493.66349 | - | - | კარგი შედეგი, მაგრამ LightGBM-ზე ოდნავ სუსტი |

---

## საბოლოო შედეგი Kaggle-ზე

Final submission გაკეთდა LightGBM მოდელით.

Kaggle Public Score: 2954.97771  
Kaggle Private Score: 3131.03223

Public და private score validation WMAE-ზე მაღალია, რაც მოსალოდნელია, რადგან Kaggle test period განსხვავდება validation split-ებისგან და მომავალ თარიღებზე forecast უფრო რთულია. მიუხედავად ამისა, LightGBM დარჩა საუკეთესო მოდელად ჩემს მიერ გატესტილ tree-based და classical მიდგომებს შორის.

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