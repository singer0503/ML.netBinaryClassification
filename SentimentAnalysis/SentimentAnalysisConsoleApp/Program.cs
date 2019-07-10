using System;
using System.IO;
using Microsoft.ML;
using SentimentAnalysisConsoleApp.DataStructures;
using Common;
using static Microsoft.ML.DataOperationsCatalog;

namespace SentimentAnalysisConsoleApp
{
    internal static class Program
    {
        private static readonly string BaseDatasetsRelativePath = @"../../../../Data";
        private static readonly string DataRelativePath = $"{BaseDatasetsRelativePath}/wikiDetoxAnnotated40kRows.tsv";

        private static readonly string DataPath = GetAbsolutePath(DataRelativePath);

        private static readonly string BaseModelsRelativePath = @"../../../../MLModels";
        private static readonly string ModelRelativePath = $"{BaseModelsRelativePath}/SentimentModel.zip";

        private static readonly string ModelPath = GetAbsolutePath(ModelRelativePath);

        static void Main(string[] args)
        {
            #region try
            // Create MLContext to be shared across the model creation workflow objects 
            // Set a random seed for repeatable/deterministic results across multiple trainings.
            var mlContext = new MLContext(seed: 1);

            //Note: 程式控制 需要訓練模型 / 載入並使用在本機的模型
            // train : 訓練模型
            // using : 使用模型
            string Status = "using"; 

            if(Status == "train"){
            #region step1to3
            // STEP 1: 載入要訓練的資料, 有表頭
            IDataView dataView = mlContext.Data.LoadFromTextFile<SentimentIssue>(DataPath, hasHeader: true);
            // 切割資料 分離出 訓練資料 / 測試資料
            TrainTestData trainTestSplit = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);
            // 訓練資料
            IDataView trainingData = trainTestSplit.TrainSet;
            // 測試資料
            IDataView testData = trainTestSplit.TestSet;

            // STEP 2: 設置通用數據流程 配置器         
            var dataProcessPipeline = mlContext.Transforms.Text.FeaturizeText(outputColumnName: "Features", inputColumnName: nameof(SentimentIssue.Text));

            // STEP 3: 設置訓練算法（BinaryClassification），然後創建並配置 modelBuilder                        
            var trainer = mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Label", featureColumnName: "Features");
            var trainingPipeline = dataProcessPipeline.Append(trainer);
            #endregion

            #region step4
            // STEP 4: 訓練模型並擬合到DataSet
            ITransformer trainedModel = trainingPipeline.Fit(trainingData);
            #endregion

            #region step5
            // STEP 5: 評估模型並顯示準確性統計數據
            var predictions = trainedModel.Transform(testData);
            var metrics = mlContext.BinaryClassification.Evaluate(data: predictions, labelColumnName: "Label", scoreColumnName: "Score");
            #endregion
            // 在主控台列印相關數據
            ConsoleHelper.PrintBinaryClassificationMetrics(trainer.ToString(), metrics);

            // STEP 6: 將訓練的模型保存/保存到.ZIP文件（MLModels資料夾下）
            mlContext.Model.Save(trainedModel, trainingData.Schema, ModelPath);

            //顯示訓練完成 model 的保存路徑
            Console.WriteLine("The model is saved to {0}", ModelPath);

            }else if (Status == "using"){
            // ！！嘗試：從.ZIP文件加載模型進行 單個測試預測＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
            // 創建一句話，先放入model
            SentimentIssue sampleStatement = new SentimentIssue { Text = "This is a very rude movie" };
            //SentimentIssue sampleStatement = new SentimentIssue { Text = "Machine learning is not fun" };
            //SentimentIssue sampleStatement = new SentimentIssue { Text = "Ancient sources for this, please." };
            //SentimentIssue sampleStatement = new SentimentIssue { Text = "This is a very fuck movie" };

            #region consume
            // 載入儲存在本機的模型
            DataViewSchema outSchema ;
            ITransformer loadModel = mlContext.Model.Load(ModelPath, out outSchema);

            // 創建與加載的訓練模型相關的預測引擎 loadModel / trainedModel
            var predEngine = mlContext.Model.CreatePredictionEngine<SentimentIssue, SentimentPrediction>(loadModel);

            // 將需要分析的資料丟入預測引擎
            var resultprediction = predEngine.Predict(sampleStatement);
            #endregion

            Console.WriteLine($"=============== Single Prediction  ===============");
            Console.WriteLine($"Text: {sampleStatement.Text} ");
            Console.WriteLine($"Prediction: {(Convert.ToBoolean(resultprediction.Prediction) ? "Toxic" : "Non Toxic")} sentiment | Probability of being toxic: {resultprediction.Probability} ");
            Console.WriteLine($"Score: {resultprediction.Score} ");
            Console.WriteLine($"================End of Process.Hit any key to exit==================================");
            Console.ReadLine();
            }
            #endregion
        }

        public static string GetAbsolutePath(string relativePath)
        {
            FileInfo _dataRoot = new FileInfo(typeof(Program).Assembly.Location);
            string assemblyFolderPath = _dataRoot.Directory.FullName;

            string fullPath = Path.Combine(assemblyFolderPath , relativePath);

            return fullPath;
        }
    }
}