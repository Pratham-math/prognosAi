import { useEffect, useState } from "react";

export default function Quiz() {
  const [questions, setQuestions] = useState([]);
  const [index, setIndex] = useState(0);
  const [loading, setLoading] = useState(true);
  const [selected, setSelected] = useState(null);
  const [score, setScore] = useState(() => Number(localStorage.getItem("quizScore")) || 0);
  const [finished, setFinished] = useState(false);
  const [dark, setDark] = useState(() => JSON.parse(localStorage.getItem("darkMode")) || false);
  const [time, setTime] = useState(() => Number(localStorage.getItem("quizTime")) || 60);

  // Fetch questions
  useEffect(() => {
    fetch("https://opentdb.com/api.php?amount=10&category=18&type=multiple")
      .then((res) => res.json())
      .then((data) => {
        const formatted = data.results.map((q) => {
          const options = [...q.incorrect_answers];
          const rand = Math.floor(Math.random() * 4);
          options.splice(rand, 0, q.correct_answer);
          return {
            question: q.question,
            correct: q.correct_answer,
            options,
          };
        });
        setQuestions(formatted);
        setLoading(false);
      });
  }, []);

  // Timer
  useEffect(() => {
    localStorage.setItem("quizTime", time);
    if (time === 0) setFinished(true);
    const timer = setInterval(() => setTime((t) => (t > 0 ? t - 1 : 0)), 1000);
    return () => clearInterval(timer);
  }, [time]);

  const toggleTheme = () => {
    setDark(!dark);
    localStorage.setItem("darkMode", JSON.stringify(!dark));
  };

  const handleNext = () => {
    if (selected === questions[index].correct) {
      const newScore = score + 1;
      setScore(newScore);
      localStorage.setItem("quizScore", newScore);
    }
    if (index === questions.length - 1) {
      setFinished(true);
      return;
    }
    setSelected(null);
    setIndex(index + 1);
  };

  const resetQuiz = () => {
    localStorage.removeItem("quizScore");
    localStorage.removeItem("quizTime");
    window.location.reload();
  };

  if (loading) return <div className="text-center p-6 text-xl">Loading...</div>;

  if (finished)
    return (
      <div className={`${dark ? "bg-gray-900 text-white" : "bg-gray-100"} min-h-screen flex items-center justify-center p-6`}>
        <div className="bg-white dark:bg-gray-800 shadow-2xl rounded-2xl p-8 text-center max-w-lg w-full">
          <h1 className="text-3xl font-bold mb-4">Quiz Finished!</h1>
          <p className="text-xl mb-6">Your Score: {score}/{questions.length}</p>
          <button
            className="px-6 py-3 bg-blue-600 hover:bg-blue-700 text-white rounded-2xl transition-colors"
            onClick={resetQuiz}
          >
            Restart Quiz
          </button>
        </div>
      </div>
    );

  const progress = ((index + 1) / questions.length) * 100;
  const timeColor = time <= 10 ? "text-red-500" : dark ? "text-white" : "text-gray-900";

  return (
    <div className={`${dark ? "bg-gray-900 text-white" : "bg-gradient-to-r from-blue-500 to-purple-600 text-white"} min-h-screen flex items-center justify-center p-4 transition-all`}>
      <div className="w-full max-w-2xl bg-white dark:bg-gray-800 rounded-3xl shadow-2xl p-6 space-y-4">
        
        {/* Header */}
        <div className="flex justify-between items-center mb-4">
          <h1 className="text-2xl font-bold">Smart Quiz</h1>
          <button
            className="px-4 py-2 bg-gray-700 hover:bg-gray-600 text-white rounded-xl transition-colors"
            onClick={toggleTheme}
          >
            {dark ? "Light" : "Dark"} Mode
          </button>
        </div>

        {/* Timer & Progress */}
        <div className="flex justify-between items-center mb-4">
          <div className={`font-medium text-lg ${timeColor}`}>⏳ Time Left: {time}s</div>
          <div className="text-sm">{index + 1}/{questions.length}</div>
        </div>
        <div className="w-full bg-gray-300 dark:bg-gray-700 h-2 rounded-full mb-4">
          <div className="h-2 bg-green-500 rounded-full transition-all" style={{ width: `${progress}%` }}></div>
        </div>

        {/* Question Card */}
        <div className="bg-gray-100 dark:bg-gray-900 rounded-2xl shadow-md p-5 space-y-4">
          <h2 className="text-lg font-semibold mb-4" dangerouslySetInnerHTML={{ __html: questions[index].question }} />

          {/* Options */}
          <div className="grid gap-3">
            {questions[index].options.map((opt, i) => {
              const isSelected = selected === opt;
              const isCorrect = opt === questions[index].correct;
              let btnClass = "w-full p-3 border rounded-xl transition-colors ";
              if (selected) {
                if (isSelected && isCorrect) btnClass += "bg-green-500 text-white";
                else if (isSelected && !isCorrect) btnClass += "bg-red-500 text-white";
                else btnClass += "bg-white dark:bg-gray-700 dark:text-white";
              } else {
                btnClass += isSelected ? "bg-blue-500 text-white" : "bg-white dark:bg-gray-700 dark:text-white hover:bg-blue-400 hover:text-white";
              }
              return (
                <button
                  key={i}
                  onClick={() => setSelected(opt)}
                  className={btnClass}
                  dangerouslySetInnerHTML={{ __html: opt }}
                  disabled={!!selected}
                />
              );
            })}
          </div>

          {/* Next Button */}
          <button
            onClick={handleNext}
            className="mt-4 w-full py-3 bg-green-600 hover:bg-green-700 text-white rounded-2xl transition-colors"
            disabled={!selected}
          >
            {index === questions.length - 1 ? "Finish" : "Next"}
          </button>
        </div>
      </div>
    </div>
  );
}
