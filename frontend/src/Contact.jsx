import { EnvelopeIcon } from "@heroicons/react/24/outline"
import { CodeBracketIcon } from "@heroicons/react/24/outline"

export default function Contact(){

return(

<div className="p-8 text-gray-300 overflow-y-auto h-full">

<h2 className="text-2xl font-semibold mb-6 text-white">
Contact
</h2>

<p className="text-sm mb-6">
Developer:
<span className="text-white font-semibold ml-2">
Sudheer Tantapureddy
</span>
</p>

{/* EMAIL */}

<div className="flex items-center gap-3 mb-5">

<EnvelopeIcon className="w-5 h-5 text-indigo-400"/>

<a
href="mailto:tsudheer1480@gmail.com"
className="text-indigo-400 hover:text-indigo-300 transition underline text-sm"
>

tsudheer1480@gmail.com

</a>

</div>

{/* GITHUB */}

<div className="flex items-center gap-3 mb-5">

<CodeBracketIcon className="w-5 h-5 text-indigo-400"/>

<a
href="https://github.com/tsudheer1480"
target="_blank"
rel="noopener noreferrer"
className="text-indigo-400 hover:text-indigo-300 transition underline text-sm"
>

github.com/tsudheer1480

</a>

</div>

{/* MESSAGE */}

<p className="text-sm text-gray-400 mt-6 leading-relaxed">

If you have questions, suggestions, or collaboration opportunities
related to AI, machine learning, or energy forecasting systems,
feel free to reach out through email or GitHub.

</p>

</div>

)

}