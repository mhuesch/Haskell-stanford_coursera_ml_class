module NeuralNet where

import qualified Data.Vector as V
import           Numeric.LinearAlgebra

data NeuralNet = NeuralNet
  { matrices :: V.Vector (Matrix Double)
  } deriving Show

matDimsToNNDims :: [(Int,Int)] -> (Int,[Int],Int)
matDimsToNNDims ds = (inputCount, hiddenCounts, outputCount)
  where
    inputCount = snd (head ds) - 1
    hiddenCounts = fst <$> init ds
    outputCount = fst (last ds)

nnDimsToMatDims :: (Int,[Int],Int) -> [(Int,Int)]
nnDimsToMatDims (inputCount, hiddenCounts, outputCount)
  = go inputCount (hiddenCounts ++ [outputCount])
  where
    go _ [] = []
    go x (y:ys) = (y, x+1) : go y ys

nnSize :: NeuralNet -> (Int,[Int],Int)
nnSize nn = matDimsToNNDims (size <$> V.toList (matrices nn))

flattenNN :: NeuralNet -> Vector Double
flattenNN nn = vjoin (map flatten (V.toList (matrices nn)))

reshapeNN :: (Int,[Int],Int) -> Vector Double -> NeuralNet
reshapeNN (inputCount, hiddenCounts, outputCount) xs = undefined

