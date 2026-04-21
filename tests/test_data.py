import unittest
from data.datasets import load_and_preprocess_dataset

class TestDataPipeline(unittest.TestCase):
    def test_load_dataset(self):
        # Using a small dataset for testing
        dataset_name = "trl-lib/Capybara"
        dataset = load_and_preprocess_dataset(dataset_name, test_size=0.1)
        
        self.assertIn("train", dataset)
        self.assertIn("test", dataset)
        self.assertTrue(len(dataset["train"]) > 0)

if __name__ == "__main__":
    unittest.main()
