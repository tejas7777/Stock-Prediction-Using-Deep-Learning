from Model import Model

model = Model();
model.CollectData('NFLX').SetTRaining().InitialiseRNN().StartTraining().PlotGraph().PredictTomorrow()


