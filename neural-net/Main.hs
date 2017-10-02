module Main where

import Control.Monad
import NeuralNet
import RandomMonad

main = exTrain

--------------------------------------------------------------------------------
-- Examples
--------------------------------------------------------------------------------

runNN :: IO ()
runNN = do
  let dims = (4,[3],2)
  nn <- randIO (randInitialNN dims)
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
      nn = mkNeuralNet [theta1, theta2]
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
  ys <- map read . words <$> readFile "data/y.txt"
  theta1 <- matrix 401 . map read . words <$> readFile "data/Theta1.txt"
  theta2 <- matrix 26 . map read . words <$> readFile "data/Theta2.txt"
  -- print $ map size [theta1,theta2,ys,xs]
  -- print (map maxElement (toColumns xs))
  -- print (map maxElement (toColumns theta1))
  -- print (map maxElement (toColumns theta2))
  let nn = mkNeuralNet [theta1, theta2]
      -- ysExpanded = matrix 10 (concatMap (mkLogicalArray 10 . round) (concat (toLists ys)))
      ysExpanded = expandLogicalArray 10 ys
  print (fst (nnCostFunction nn xs ysExpanded 0))
  putStrLn "0.287629 is expected"
  print (fst (nnCostFunction nn xs ysExpanded 1))
  putStrLn "0.383770 is expected"

exTrain :: IO ()
exTrain = do
  xs <- matrix 400 . map read . words <$> readFile "data/X.txt"
  ys <- map read . words <$> readFile "data/y.txt"
  let dims = (400, [25], 10)
  initNN <- randIO (randInitialNN dims)
  let initCost = fst (nnCostFunction initNN xs ysExpanded lambda)
      lambda = 1
      -- ysExpanded = matrix 10 (concatMap (mkLogicalArray 10 . round) (concat (toLists ys)))
      ysExpanded = expandLogicalArray 10 ys
      f (count, costPairs, nn) iter =
        let nn' = trainNN iter lambda nn xs ysExpanded
            count' = count+iter
            cost = fst (nnCostFunction nn' xs ysExpanded lambda)
            pair = (count', cost)
         in (count', costPairs ++ [pair], nn')
      (_, costPairs, finalNN) = foldl f (0, [(0,initCost)], initNN) (replicate 70 50)
  forM_ costPairs $ \(iters,cost) ->
    putStrLn $ "iterations: " ++ show iters ++ ", cost: " ++ show cost
