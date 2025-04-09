import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import when, col
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, ChiSqSelector
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml import Pipeline

# Ensure output folder exists
os.makedirs("output", exist_ok=True)

# Start Spark session
spark = SparkSession.builder.appName("CustomerChurnMLlib").getOrCreate()

# Load dataset
data_path = "customer_churn.csv"
df = spark.read.csv(data_path, header=True, inferSchema=True)

# =============================
# Task 1: Data Preprocessing
# =============================
def preprocess_data(df):
    print("===== TASK 1: Preprocessing Data =====")

    df = df.withColumn("TotalCharges", when(col("TotalCharges").isNull(), 0).otherwise(col("TotalCharges")))

    categorical_cols = ["gender", "PhoneService", "InternetService"]
    indexers = [StringIndexer(inputCol=c, outputCol=c + "_Index") for c in categorical_cols]
    encoders = [OneHotEncoder(inputCol=c + "_Index", outputCol=c + "_Vec") for c in categorical_cols]

    numerical_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    assembler_inputs = [c + "_Vec" for c in categorical_cols] + numerical_cols
    assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features")

    pipeline = Pipeline(stages=indexers + encoders + [assembler])
    model = pipeline.fit(df)
    final_df = model.transform(df)

    final_df = final_df.withColumn("ChurnIndex", when(col("Churn") == "Yes", 1.0).otherwise(0.0))
    final_output = final_df.select("features", "ChurnIndex")

    with open("output/task1_output.txt", "w") as f:
        f.write("===== Task 1: Features with ChurnIndex =====\n")
        f.write(final_output._jdf.showString(5, 20, False))

    return final_output

# =============================
# Task 2: Logistic Regression
# =============================
def train_logistic_regression_model(df):
    print("\n===== TASK 2: Logistic Regression =====")

    train, test = df.randomSplit([0.8, 0.2], seed=42)
    lr = LogisticRegression(featuresCol="features", labelCol="ChurnIndex")
    model = lr.fit(train)
    predictions = model.transform(test)

    evaluator = BinaryClassificationEvaluator(labelCol="ChurnIndex")
    auc = evaluator.evaluate(predictions)

    with open("output/task2_output.txt", "w") as f:
        f.write("===== Logistic Regression AUC =====\n")
        f.write(f"AUC: {auc:.4f}\n")

# =============================
# Task 3: Chi-Square Selector
# =============================
def feature_selection(df):
    print("\n===== TASK 3: Feature Selection =====")

    selector = ChiSqSelector(numTopFeatures=5, featuresCol="features", outputCol="selectedFeatures", labelCol="ChurnIndex")
    result = selector.fit(df).transform(df)
    selected_output = result.select("selectedFeatures", "ChurnIndex")

    with open("output/task3_output.txt", "w") as f:
        f.write("===== Top 5 Selected Features =====\n")
        f.write(selected_output._jdf.showString(5, 20, False))

# =============================
# Task 4: Hyperparameter Tuning
# =============================
def tune_and_compare_models(df):
    print("\n===== TASK 4: Model Comparison =====")

    train, test = df.randomSplit([0.8, 0.2], seed=42)
    evaluator = BinaryClassificationEvaluator(labelCol="ChurnIndex")

    models_with_params = [
        {
            "name": "LogisticRegression",
            "model": LogisticRegression(labelCol="ChurnIndex"),
            "paramGrid": ParamGridBuilder()
                .addGrid(LogisticRegression().regParam, [0.01, 0.1])
                .addGrid(LogisticRegression().maxIter, [10, 20])
                .build(),
            "extract_params": lambda m: f"regParam={m.getRegParam()}, maxIter={m.getMaxIter()}"
        },
        {
            "name": "DecisionTree",
            "model": DecisionTreeClassifier(labelCol="ChurnIndex"),
            "paramGrid": ParamGridBuilder()
                .addGrid(DecisionTreeClassifier().maxDepth, [5, 10])
                .build(),
            "extract_params": lambda m: f"maxDepth={m.getMaxDepth()}"
        },
        {
            "name": "RandomForest",
            "model": RandomForestClassifier(labelCol="ChurnIndex"),
            "paramGrid": ParamGridBuilder()
                .addGrid(RandomForestClassifier().maxDepth, [10, 15])
                .addGrid(RandomForestClassifier().numTrees, [20, 50])
                .build(),
            "extract_params": lambda m: f"maxDepth={m.getMaxDepth()}, numTrees={m.getNumTrees}"
        },
        {
            "name": "GBT",
            "model": GBTClassifier(labelCol="ChurnIndex"),
            "paramGrid": ParamGridBuilder()
                .addGrid(GBTClassifier().maxDepth, [5, 10])
                .addGrid(GBTClassifier().maxIter, [10, 20])
                .build(),
            "extract_params": lambda m: f"maxDepth={m.getMaxDepth()}, maxIter={m.getMaxIter()}"
        }
    ]

    with open("output/task4_output.txt", "w") as f:
        f.write("===== Cross-Validation Model Comparison =====\n")
        for entry in models_with_params:
            name = entry["name"]
            model = entry["model"]
            paramGrid = entry["paramGrid"]
            extract_params = entry["extract_params"]

            print(f"Tuning {name}...")
            f.write(f"\nTuning {name}...\n")

            cv = CrossValidator(estimator=model,
                                estimatorParamMaps=paramGrid,
                                evaluator=evaluator,
                                numFolds=5)
            cv_model = cv.fit(train)
            auc = evaluator.evaluate(cv_model.transform(test))
            best_model = cv_model.bestModel

            f.write(f"{name} Best Model Accuracy (AUC): {auc:.2f}\n")
            f.write(f"Best Params for {name}: {extract_params(best_model)}\n")

# =============================
# Run all tasks
# =============================
preprocessed_df = preprocess_data(df)
train_logistic_regression_model(preprocessed_df)
feature_selection(preprocessed_df)
tune_and_compare_models(preprocessed_df)

# Stop Spark
spark.stop()