import unittest
from datetime import datetime
from voice_assistant_commands import live_answer, file_query


class VoiceAssistantCommandTests(unittest.TestCase):
    def test_time_and_date_use_live_local_values(self):
        kind, answer = live_answer("what time is it")
        self.assertEqual(kind, "time")
        self.assertTrue(answer.startswith("It is "))
        self.assertEqual(live_answer("what's the time")[0], "time")
        kind, answer = live_answer("what is the date")
        self.assertEqual(kind, "date")
        self.assertIn(str(datetime.now().year), answer)

    def test_system_information(self):
        result = live_answer("give me system info")
        self.assertEqual(result[0], "system")
        self.assertIn("running", result[1])

    def test_file_queries(self):
        self.assertEqual(file_query("find file named budget.xlsx"), "budget.xlsx")
        self.assertEqual(file_query("search files for holiday photos"), "holiday photos")
        self.assertIsNone(file_query("search files"))

    def test_unrelated_question_is_not_live_command(self):
        self.assertIsNone(live_answer("why is the sky blue"))


if __name__ == "__main__":
    unittest.main()
