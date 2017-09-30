module NeuralNetSpec where

import           Test.Hspec
import           Test.Hspec.QuickCheck (modifyMaxSuccess)
import           Test.QuickCheck.Property

import NeuralNet

spec :: Spec
spec = do
  modifyMaxSuccess (const 2000) $
    it "matDimsToNNDims . nnDimsToMatDims == id" $
      property $ forAll nnDimsGen $ \dim -> dim == (matDimsToNNDims (nnDimsToMatDims dim))

  modifyMaxSuccess (const 100) $
    it "reshapeNN . flattenNN == id" $
      property $ forAll nnGen $ \nn -> nn == (reshapeNN (nnSize nn) (flattenNN nn))
