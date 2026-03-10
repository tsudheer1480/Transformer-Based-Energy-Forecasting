import { Link } from "react-router-dom"

export default function About(){

return(

<div className="min-h-screen bg-gray-900 text-gray-200 p-10">

<h1 className="text-3xl font-bold mb-6">
About This Project
</h1>

<p className="text-gray-400 max-w-3xl">

This dashboard forecasts electricity demand using a Transformer-based deep learning model.

Users can upload historical energy datasets and generate predictions for:

• 24 hours  
• 7 days  
• 30 days  

The system also provides explainable AI insights to understand how predictions are made.

</p>

<Link
to="/"
className="inline-block mt-6 text-indigo-400 hover:text-indigo-300"
>

← Back to Dashboard

</Link>

</div>

)

}