from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile

import pandas as pd
from six.moves import urllib
import tensorflow as tf


CSV_COLUMNS = [
    "Neo_Reference_ID",  "Name",  "Absolute_Magnitude",  "Est_Dia_in_M_min", "Est_Dia_in_M_max", "Epoch_Date_Close_Approach", "Relative_Velocity_km_per_sec",  "Miss_Dist_kilometers", "Orbit_ID",  "Orbit_Uncertainity",  "Minimum_Orbit_Intersection",  "Jupiter_Tisserand_Invariant", "Epoch_Osculation",  "Eccentricity",  "Semi_Major_Axis", "Inclination", "Asc_Node_Longitude",  "Orbital_Period",  "Perihelion_Distance", "Perihelion_Arg",  "Aphelion_Dist", "Perihelion_Time", "Mean_Anomaly",  "Mean_Motion", "Hazardous"

]


# Continuous base columns.
Absolute_Magnitude = tf.feature_column.numeric_column("Absolute_Magnitude")
Est_Dia_in_M_min = tf.feature_column.numeric_column("Est_Dia_in_M_min")
Epoch_Date_Close_Approach = tf.feature_column.numeric_column("Epoch_Date_Close_Approach")
Relative_Velocity_km_per_sec = tf.feature_column.numeric_column("Relative_Velocity_km_per_sec")
Miss_Dist_Kilometers= tf.feature_column.numeric_column("Miss_Dist_kilometers")
Orbit_ID = tf.feature_column.numeric_column("Orbit_ID")
Orbit_Uncertainity = tf.feature_column.numeric_column("Orbit_Uncertainity")
Minimum_Orbit_Intersection = tf.feature_column.numeric_column("Minimum_Orbit_Intersection")
Jupiter_Tisserand_Invariant = tf.feature_column.numeric_column("Jupiter_Tisserand_Invariant")
Epoch_Osculation = tf.feature_column.numeric_column("Epoch_Osculation")
Eccentricity = tf.feature_column.numeric_column("Eccentricity")
Semi_Major_Axis = tf.feature_column.numeric_column("Semi_Major_Axis")
Asc_Node_Longitude = tf.feature_column.numeric_column("Asc_Node_Longitude")
Orbital_Period = tf.feature_column.numeric_column("Orbital_Period")
Perihelion_Distance = tf.feature_column.numeric_column("Perihelion_Distance")
Perihelion_Arg = tf.feature_column.numeric_column("Perihelion_Arg")
Aphelion_Dist = tf.feature_column.numeric_column("Aphelion_Dist")
Perihelion_Time = tf.feature_column.numeric_column("Perihelion_Time")
Mean_Anomaly = tf.feature_column.numeric_column("Mean_Anomaly")
Mean_Motion = tf.feature_column.numeric_column("Mean_Motion")

deep_columns = [
    Absolute_Magnitude,
    Est_Dia_in_M_min,
    Epoch_Date_Close_Approach,
    Relative_Velocity_km_per_sec,
    Miss_Dist_Kilometers,
    Orbit_ID,
    Orbit_Uncertainity,
    Minimum_Orbit_Intersection,
    Jupiter_Tisserand_Invariant,
    Epoch_Osculation,
    Eccentricity,
    Semi_Major_Axis,
    Asc_Node_Longitude,
    Orbital_Period,
    Perihelion_Distance,
    Perihelion_Arg,
    Aphelion_Dist,
    Perihelion_Time,
    Mean_Anomaly,
    Mean_Motion

]

def maybe_download(train_data, test_data):
  """Maybe downloads training data and returns train and test file names."""
  if train_data:
    train_file_name = train_data
  else:
    train_file = tempfile.NamedTemporaryFile(delete=False)
    urllib.request.urlretrieve(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
        train_file.name)  # pylint: disable=line-too-long
    train_file_name = train_file.name
    train_file.close()
    print("Training data is downloaded to %s" % train_file_name)

  if test_data:
    test_file_name = test_data
  else:
    test_file = tempfile.NamedTemporaryFile(delete=False)
    urllib.request.urlretrieve(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test",
        test_file.name)  # pylint: disable=line-too-long
    test_file_name = test_file.name
    test_file.close()
    print("Test data is downloaded to %s"% test_file_name)

  return train_file_name, test_file_name


def build_estimator(model_dir, model_type):
  """Build an estimator."""
  if model_type == "wide":
    m = tf.estimator.LinearClassifier(
        model_dir=model_dir, feature_columns=base_columns + crossed_columns)
  elif model_type == "deep":
    m = tf.estimator.DNNClassifier(
        model_dir=model_dir,
        feature_columns=deep_columns,
        hidden_units=[100, 50])
  else:
    m = tf.estimator.DNNLinearCombinedClassifier(
        model_dir=model_dir,
        linear_feature_columns=crossed_columns,
        dnn_feature_columns=deep_columns,
        dnn_hidden_units=[100, 50])
  return m


def input_fn(data_file, num_epochs, shuffle):
  """Input builder function."""
  df_data = pd.read_csv(
      tf.gfile.Open(data_file),
      names=CSV_COLUMNS,
      skipinitialspace=True,
      engine="python",
      skiprows=1)
  # remove NaN elements
  df_data = df_data.dropna(how="any", axis=0)
  labels = df_data["Hazardous"].apply(lambda x: x).astype(int)
  print (labels)
  #print(labels)
  return tf.estimator.inputs.pandas_input_fn(
      x=df_data,
      y=labels,
      batch_size=100,
      num_epochs=num_epochs,
      shuffle=shuffle,
      num_threads=5)


def train_and_eval(model_dir, model_type, train_steps, train_data, test_data):
  """Train and evaluate the model."""
  train_file_name, test_file_name = maybe_download(train_data, test_data)
  model_dir = tempfile.mkdtemp() if not model_dir else model_dir

  m = build_estimator(model_dir, model_type)
  # set num_epochs to None to get infinite stream of data.
  m.train(
      input_fn=input_fn(train_file_name, num_epochs=None, shuffle=True),
      steps=train_steps)
  # set steps to None to run evaluation until all data consumed.
  results = m.evaluate(
      input_fn=input_fn(test_file_name, num_epochs=1, shuffle=False),
      steps=None)
  print("model directory = %s" % model_dir)
  for key in sorted(results):
    print("%s: %s" % (key, results[key]))


FLAGS = None


def main(_):
  train_and_eval(FLAGS.model_dir, FLAGS.model_type, FLAGS.train_steps,
                 FLAGS.train_data, FLAGS.test_data)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")
  parser.add_argument(
      "--model_dir",
      type=str,
      default="",
      help="Base directory for output models."
  )
  parser.add_argument(
      "--model_type",
      type=str,
      default="deep",
      help="Valid model types: {'wide', 'deep', 'wide_n_deep'}."
  )
  parser.add_argument(
      "--train_steps",
      type=int,
      default=2000,
      help="Number of training steps."
  )
  parser.add_argument(
      "--train_data",
      type=str,
      default="names.csv",
      help="Path to the training data."
  )
  parser.add_argument(
      "--test_data",
      type=str,
      default="test.csv",
      help="Path to the test data."
  )
  FLAGS, unparsed = parser.parse_known_args()
tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)