module NeuralNetSpec where

import Control.Monad
import Test.Hspec
import Test.Hspec.QuickCheck (modifyMaxSuccess)
import Test.QuickCheck.Arbitrary
import Test.QuickCheck.Gen
import Test.QuickCheck.Property

import NeuralNet

spec :: Spec
spec = do
  modifyMaxSuccess (const 1000) $
    it "matDimsToNNDims . nnDimsToMatDims == id" $
      property $ forAll nnDimsGen $ \dim -> dim == (matDimsToNNDims (nnDimsToMatDims dim))

positiveInt :: Gen Int
positiveInt = suchThat arbitrary (> 0)

nnDimsGen :: Gen (Int,[Int],Int)
nnDimsGen = (>**<) positiveInt (listOf positiveInt) positiveInt



-- Lifted from https://hackage.haskell.org/package/checkers-0.4.7/docs/src/Test-QuickCheck-Instances-Tuple.html#%3E%2A%3C
{- | Generates a 2-tuple using its arguments to generate the parts.
-}
(>*<) :: Gen a -> Gen b -> Gen (a,b)
x >*< y = liftM2 (,) x y

{- | Generates a 3-tuple using its arguments to generate the parts.
-}
(>**<) :: Gen a -> Gen b -> Gen c -> Gen (a,b,c)
(>**<) x y z = liftM3 (,,) x y z
