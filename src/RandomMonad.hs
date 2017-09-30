module RandomMonad where

import qualified Control.Monad.State as S
import           System.Random

type Rand a = S.State StdGen a

runRand :: StdGen -> Rand a -> (a, StdGen)
runRand g m = S.runState m g

evalRand :: StdGen -> Rand a -> a
evalRand g m = S.evalState m g

randRandom :: (Random a) => Rand a
randRandom = randomOp random

randUniform :: (Random a) => a -> a -> Rand a
randUniform lo hi = randomOp (randomR (lo,hi))

randomElem :: [a] -> Rand a
randomElem ls = do
  idx <- randomOp (randomR (0, (length ls) - 1))
  return (ls !! idx)

randomOp :: (StdGen -> (a, StdGen)) -> Rand a
randomOp op = do
  g <- S.get
  let (x, g') = op g
  S.put g'
  return x
