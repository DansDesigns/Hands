import unittest
from streamtts_python import TextSegmenter


class TextSegmenterTests(unittest.TestCase):
    def test_fragments_split_inside_words(self):
        s = TextSegmenter("balanced")
        self.assertEqual(s.push("Hel"), [])
        self.assertEqual(s.push("lo wor"), [])
        self.assertEqual(s.push("ld "), ["Hello world"])

    def test_balanced_groups_following_words(self):
        s = TextSegmenter("balanced")
        self.assertEqual(s.push("one two three four five six "), ["one two", "three four five"])
        self.assertEqual(s.flush(), ["six"])

    def test_sentence_punctuation_flushes_early(self):
        s = TextSegmenter("balanced")
        self.assertEqual(s.push("Hi!"), ["Hi!"])

    def test_flushes_unterminated_last_word(self):
        s = TextSegmenter("balanced")
        self.assertEqual(s.push("hello"), [])
        self.assertEqual(s.flush(), ["hello"])

    def test_clause_punctuation_after_two_words(self):
        s = TextSegmenter("balanced")
        self.assertEqual(s.push("these two, more words "), ["these two,"])
        self.assertEqual(s.flush(), ["more words"])


if __name__ == "__main__":
    unittest.main()
