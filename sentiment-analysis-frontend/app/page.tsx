'use client';
import { useState, FormEvent, ChangeEvent } from "react";

interface PredictionSuccess {
  predicted_label: number;
  confidence: number;
}

interface PredictionError {
  error: string;
}

type PredictionResult = PredictionSuccess | PredictionError;

function isPredictionError(result: PredictionResult): result is PredictionError {
  return 'error' in result;
}

export default function Home() {
  const [review, setReview] = useState<string>("");
  const [touched, setTouched] = useState<boolean>(false);
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState<boolean>(false);

  const isReviewEmpty = review.trim().length === 0;
  const showError = touched && isReviewEmpty;

  const handleEvaluate = async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (isReviewEmpty) return;
    
    setLoading(true);
    try {
      const response = await fetch("http://localhost:8001/api/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ text: review }),
      });

      const data = await response.json();
      if (response.ok) {
        setPrediction(data as PredictionSuccess);
      } else {
        setPrediction({ error: data.detail || "An error occurred" });
      }
    } catch (error) {
      setPrediction({ error: "Failed to fetch prediction" });
    } finally {
      setLoading(false);
    }
  };

  const handleInputChange = (e: ChangeEvent<HTMLTextAreaElement>) => {
    setReview(e.target.value);
    setTouched(true);
  };

  const handleReset = () => {
    setReview("");
    setTouched(false);
    setPrediction(null);
    setLoading(false);
  };

  return (
    <div className="min-h-screen bg-[#0A1929]">
      <div className="p-8 max-w-2xl mx-auto">
        {/* Welcome Header Section */}
        <div className="text-center mb-8 p-6 bg-[#0F2942] rounded-lg border border-[#1E4976]">
          <h1 className="text-3xl font-bold mb-3 text-white">
            Restaurant Review Sentiment Analysis
          </h1>
          <p className="text-gray-300 text-lg">
            Welcome to Use the Review Sentiment Analysis Interface
          </p>
        </div>

        <form onSubmit={handleEvaluate} className="space-y-4">
          <label 
            htmlFor="review" 
            className="block mb-2 text-lg text-gray-200"
          >
            Enter your restaurant review:
          </label>
          <div className="space-y-1">
            <textarea
              id="review"
              value={review}
              onChange={handleInputChange}
              onBlur={() => setTouched(true)}
              rows={4}
              className={`w-full p-4 rounded bg-[#0F2942] text-white placeholder-gray-400 border ${
                showError ? 'border-red-500' : 'border-[#1E4976]'
              } focus:outline-none focus:ring-2 ${
                showError ? 'focus:ring-red-500' : 'focus:ring-[#2977CC]'
              }`}
              placeholder="Type your review here..."
            />
            {showError && (
              <p className="text-red-500 text-sm">*The review can't be empty</p>
            )}
          </div>
          <div className="flex space-x-4">
            <button 
              type="submit" 
              className="px-6 py-3 bg-[#2977CC] text-white rounded hover:bg-[#3688DE] disabled:bg-[#1E4976] disabled:cursor-not-allowed transition duration-200"
              disabled={loading || isReviewEmpty}
            >
              {loading ? "Evaluating..." : "Evaluate"}
            </button>
            <button 
              type="button"
              onClick={handleReset}
              className="px-6 py-3 bg-[#1E4976] text-white rounded hover:bg-[#2B5A8B] transition duration-200"
            >
              Reset
            </button>
          </div>
        </form>

        {prediction && (
          <div className="mt-8 p-6 bg-[#0F2942] rounded-lg border border-[#1E4976]">
            <h2 className="text-xl font-bold mb-4 text-white">Evaluation Result:</h2>
            {isPredictionError(prediction) ? (
              <p className="text-red-400">{prediction.error}</p>
            ) : (
              <div className="space-y-2">
                <p>
                  <strong className="text-gray-200">Sentiment:</strong>{' '}
                  <span className={prediction.predicted_label === 1 ? "text-green-400" : "text-red-400"}>
                    {prediction.predicted_label === 1 ? "Positive" : "Negative"}
                  </span>
                </p>
                <p>
                  <strong className="text-gray-200">Confidence:</strong>{' '}
                  <span className="text-[#63B2F7]">
                    {(prediction.confidence * 100).toFixed(2)}%
                  </span>
                </p>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}