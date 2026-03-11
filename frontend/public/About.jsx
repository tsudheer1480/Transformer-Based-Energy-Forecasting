import { Disclosure, Transition } from "@headlessui/react"

export default function About(){

const sections = [

{
title:"Project Overview",
content:`
The AI Energy Load Forecast Dashboard predicts electricity demand using
a Transformer-based deep learning model.

Users can upload historical electricity datasets and generate
multi-horizon forecasts including:

• 24 Hour Forecast
• 7 Day Forecast
• 30 Day Forecast

The system also provides explainable AI insights to help users
understand how predictions are generated.
`
},

{
title:"Key Features",
content:`
• Upload historical electricity datasets in CSV format

• Multi-horizon forecasting (24H / 7D / 30D)

• Interactive forecast tables with sorting and search

• Forecast visualization through graphs

• Model evaluation metrics (MAE and error %)

• Explainable AI insights including attention analysis
`
},

{
title:"AI Model",
content:`
The forecasting engine uses a Transformer-based deep learning
architecture designed for time-series prediction.

Transformer models use attention mechanisms to capture long-range
temporal dependencies in sequential data. This allows the model
to learn complex electricity consumption patterns such as:

• Daily demand cycles
• Weekly patterns
• Weather influence on load
• Historical load dependencies
`
},

{
title:"Dataset Requirements",
content:`
Minimum Dataset Size

• The dataset must contain at least 1600 rows
• Data must be sequential (time ordered)
• File must be uploaded in CSV format


Required Columns

time
load
solar
wind
wind_onshore
wind_offshore
hour
day_of_week
day_of_month
month
is_weekend
is_holiday
load_forecast
temperature
humidity
wind_speed
precipitation
weather_code
temperature_lag_1
humidity_lag_1
temperature_lag_24
humidity_lag_24
load_lag_1
load_lag_24
load_lag_168
rolling_mean_24
rolling_std_24


Column Description

time → timestamp of the observation

load → electricity demand value

solar → solar energy generation

wind → total wind energy production

wind_onshore → onshore wind generation

wind_offshore → offshore wind generation

temperature → ambient temperature

humidity → atmospheric humidity

wind_speed → wind speed measurement

precipitation → rainfall level

weather_code → encoded weather condition

hour / day_of_week / month → time features used by the model

lag features (load_lag_*, temperature_lag_*, humidity_lag_*)
→ previous observations used for forecasting context

rolling statistics (rolling_mean_24, rolling_std_24)
→ short term load trends used by the model


Important Notes

• Dataset must not contain missing timestamps

• Rows must be sorted chronologically

• Missing values should be cleaned before upload
`
},

{
title:"Technology Stack",
content:`
Frontend
React.js + TailwindCSS

Backend
Python FastAPI

Machine Learning
PyTorch Transformer model

Visualization
Interactive forecast graphs
`

},

{
title:"Developer",
content:`
Developed by Sudheer Tantapureddy.

Focused on building AI-driven systems for solving
real-world problems such as energy forecasting
and intelligent infrastructure.
`
}

]

return(

<div className="p-8 text-gray-300 overflow-y-auto h-full">

<h2 className="text-2xl font-semibold mb-6 text-white">
About This Project
</h2>

<div className="space-y-3">

{sections.map((section,i)=>(

<Disclosure key={i}>

<div className="bg-gray-800/50 backdrop-blur-md rounded-lg hover:bg-gray-700 transition-colors overflow-hidden">

<Disclosure.Button className="w-full px-4 py-3 text-left text-sm font-semibold hover:text-indigo-400 transition">

{section.title}

</Disclosure.Button>

<Transition
enter="transition-all duration-500 ease-out"
enterFrom="opacity-0 -translate-y-3 scale-95"
enterTo="opacity-100 translate-y-0 scale-100"
leave="transition-all duration-300 ease-in"
leaveFrom="opacity-100 translate-y-0 scale-100"
leaveTo="opacity-0 -translate-y-2 scale-95"
>

<Disclosure.Panel className="px-5 pb-4 text-sm leading-relaxed text-gray-300 whitespace-pre-line">

{section.content}

{/* Download dataset button */}
{section.title === "Dataset Requirements" && (
<a
href="/sample_dataset.csv"
className="mt-4 inline-block bg-indigo-600 px-3 py-1 text-xs rounded hover:bg-indigo-700 transition"
>
Download Sample Dataset
</a>
)}

</Disclosure.Panel>

</Transition>

</div>

</Disclosure>

))}

</div>

</div>

)

}