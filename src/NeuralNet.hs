{-# LANGUAGE FlexibleContexts #-}
module NeuralNet where

import           Control.Monad
import qualified Data.Vector as V
import           Numeric.LinearAlgebra
import           Test.QuickCheck.Arbitrary (arbitrary)
import           Test.QuickCheck.Gen

data NeuralNet = NeuralNet
  { matrices :: V.Vector (Matrix Double)
  } deriving (Eq, Show)

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
reshapeNN dims xs = NeuralNet (V.fromList mats)
  where
    mats = zipWith reshape (map snd matDims) subVecs

    sv i c = subVector i c xs
    subVecs = zipWith sv idxs counts

    matDims = nnDimsToMatDims dims
    counts = map (\(r,c) -> r*c) matDims
    idxs = scanl (+) 0 counts


-- Returns the cost, and a neural net of gradients
nnCostFunction :: NeuralNet -> Matrix R -> Matrix R -> Double
               -> (Double, NeuralNet)
nnCostFunction nn xs ys lambda = (jCost, gradients)
  where
    m = rows xs
    aN = forwardProp xs (matrices nn)

    jMatrix = (-ys * log aN) - ((1-ys) * log (1-aN))
    jCost = sumElements jMatrix / (fromIntegral m)

    gradients = nn

forwardProp :: Matrix R -> V.Vector (Matrix R) -> Matrix R
forwardProp = V.foldl f
  where
    f acc mat = sigmoid $ (1 ||| acc) <> tr mat

    -- yExpanded = matrix (concatMap (mkLogicalArray numLabels . round) (toList (unwrap ys))) :: L m numLabels

sigmoid x = 1 / (1 + exp(-x))
sigGrad x = (sigmoid x) * (1 - (sigmoid x))

exNN :: NeuralNet
exNN = reshapeNN (4,[3],2) (vector (replicate 23 1))

xor_xnor :: IO ()
xor_xnor = do
  let theta1 = matrix 3 [ -30, 20, 20 -- AND
                        , -10, 20, 20 -- OR
                        ]
      theta2 = matrix 3 [ -10, -20,  20 -- XOR
                        ,  10,  20, -20 -- XNOR
                        ]
      nn = NeuralNet (V.fromList [theta1, theta2])
      xs = matrix 2 [ 1, 1
                    , 0, 1
                    , 1, 0
                    , 0, 0
                    ]
      -- the numbers represent the index of the "correct" output node
      ys = matrix 2 [ 0, 1
                    , 1, 0
                    , 1, 0
                    , 0, 1
                    ]
      lambda = 0
      (cost, grad) = nnCostFunction nn xs ys lambda
  print cost
  putStrLn "9.09215500328088e-05 is expected"

--------------------------------------------------------------------------------
-- Instances
--------------------------------------------------------------------------------
positiveInt :: Gen Int
positiveInt = suchThat arbitrary (> 0)

nnDimsGen :: Gen (Int,[Int],Int)
nnDimsGen = (>**<) positiveInt (listOf positiveInt) positiveInt

nnGen :: Gen NeuralNet
nnGen = nnDimsGen >>= nnWithDimsGen

nnWithDimsGen :: (Int, [Int], Int) -> Gen NeuralNet
nnWithDimsGen dims = do
  let matDims = nnDimsToMatDims dims
  mats <- mapM (uncurry matrixGen) matDims
  return (NeuralNet (V.fromList mats))

matrixGen :: Int -> Int -> Gen (Matrix R)
matrixGen r c = do
  xs <- replicateM (r*c) arbitrary
  return (matrix c xs)

-- Lifted from https://hackage.haskell.org/package/checkers-0.4.7/docs/src/Test-QuickCheck-Instances-Tuple.html#%3E%2A%3C
{- | Generates a 2-tuple using its arguments to generate the parts.
-}
(>*<) :: Gen a -> Gen b -> Gen (a,b)
x >*< y = liftM2 (,) x y

{- | Generates a 3-tuple using its arguments to generate the parts.
-}
(>**<) :: Gen a -> Gen b -> Gen c -> Gen (a,b,c)
(>**<) x y z = liftM3 (,,) x y z
