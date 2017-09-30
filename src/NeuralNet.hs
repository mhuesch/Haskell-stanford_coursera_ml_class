{-# LANGUAGE FlexibleContexts #-}
module NeuralNet where

import           Control.Monad
import qualified Data.Vector as V
import           Numeric.LinearAlgebra
import           System.Random
import           Test.QuickCheck.Arbitrary (arbitrary)
import           Test.QuickCheck.Gen

import RandomMonad

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
nnCostFunction nn xs ys lambda = (jCost, gradNN)
  where
    m = fromIntegral (rows xs)
    mats = matrices nn
    (activations, linearSums) = feedForward xs mats
    activationN = head activations

    jMatrix = (-ys * log activationN) - ((1-ys) * log (1-activationN))
    regVal = let drop1SqSum mat = sumElements ((dropColumns 1 mat) ** 2)
                 sqSum = V.sum (V.map drop1SqSum mats)
              in sqSum * lambda / 2
    jCost = (sumElements jMatrix + regVal) / m

    deltaN = activationN - ys
    -- We start with the Nth delta in our accumulator and work backwards
    -- We drop the first linearSum because that corresponds to layer N and we are
    -- working with the previous layer.
    deltas = reverse $ foldl computeDelta [deltaN] (tail (zip (V.toList mats) linearSums))
    computeDelta ds@(dNext:_) (theta, lSumPrev) = let dCurrent = dNext <> (dropColumns 1 theta) * sigGrad lSumPrev
                                                   in dCurrent:ds

    rawGrads = reverse $ zipWith (\del act -> (tr del <> (1 ||| act))) deltas (tail activations)
    -- we only regularize the non-bias terms
    regularizers = map (\mat -> scalar lambda * (0 ||| dropColumns 1 mat)) (V.toList mats)
    grads = zipWith (\g r -> (g + r) / scalar m) rawGrads regularizers
    gradNN = NeuralNet (V.fromList grads)

-- returns the final activation and progressive linear combinations (in reverse order)
feedForward :: Matrix R -> V.Vector (Matrix R) -> ([Matrix R], [Matrix R])
feedForward xs mats = V.foldl f ([xs], []) mats
  where
    f (as@(aPrev:_), ls) mat = let lCurrent = (1 ||| aPrev) <> tr mat
                                in ((sigmoid lCurrent):as, lCurrent:ls)

sigmoid x = 1 / (1 + exp(-x))
sigGrad x = (sigmoid x) * (1 - (sigmoid x))



randInitialNN :: (Int, [Int], Int) -> Rand NeuralNet
randInitialNN dims = do
  let matDims = nnDimsToMatDims dims
  mats <- mapM (uncurry randEpsilonMat) matDims
  return (NeuralNet (V.fromList mats))

randEpsilonMat :: Int -> Int -> Rand (Matrix R)
randEpsilonMat r c = do
  seed <- randRandom
  let mat = uniformSample seed r (replicate r (0,1))
      l_in = r - 1
      l_out = c
      epsilonInit = sqrt 6 / sqrt (fromIntegral (l_in * l_out))
  return ((mat * 2 * epsilonInit) - epsilonInit)

runNN :: IO ()
runNN = do
  g <- getStdGen
  let dims = (4,[3],2)
      nn = evalRand g (randInitialNN dims)
  print nn

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
      ps = [ (0, "9.09215500328088e-05")
           , (0.5, "2.00000090921550e+02")
           , (1, "4.00000090921550e+02")
           ]
  forM_ ps $ \(lambda, expected) -> do
    let (cost, grad) = nnCostFunction nn xs ys lambda
    putStrLn $ replicate 40 '-'
    putStrLn $ "lambda = " ++ show lambda
    print cost
    putStrLn $ expected ++ " is expected"
    print grad
    putStrLn $ replicate 40 '-' ++ "\n"

exFeedForward :: IO ()
exFeedForward = do
  xs <- matrix 400 . map read . words <$> readFile "data/X.txt"
  ys <- matrix 1 . map read . words <$> readFile "data/y.txt"
  theta1 <- matrix 401 . map read . words <$> readFile "data/Theta1.txt"
  theta2 <- matrix 26 . map read . words <$> readFile "data/Theta2.txt"
  -- print $ map size [theta1,theta2,ys,xs]
  -- print (map maxElement (toColumns xs))
  -- print (map maxElement (toColumns theta1))
  -- print (map maxElement (toColumns theta2))
  let nn = NeuralNet (V.fromList [theta1, theta2])
      ysExpanded = matrix 10 (concatMap (mkLogicalArray 10 . round) (concat (toLists ys)))
  print (fst (nnCostFunction nn xs ysExpanded 0))
  putStrLn "0.287629 is expected"
  print (fst (nnCostFunction nn xs ysExpanded 1))
  putStrLn "0.383770 is expected"

mkLogicalArray :: Num a => Int -> Int -> [a]
mkLogicalArray len pos = take len (replicate frontLen 0 ++ [1] ++ repeat 0)
  where
    frontLen = pos - 1

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
