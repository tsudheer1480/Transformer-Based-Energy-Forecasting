import { Link } from "react-router-dom"

export default function Contact(){

return(

<div className="min-h-screen bg-gray-900 text-gray-200 p-10">

<h1 className="text-3xl font-bold mb-6">
Contact
</h1>

<p className="text-gray-400">

Developer: <b>Sudheer Tantapureddy</b>

</p>

<p className="text-gray-400 mt-2">
Email: yourmail@example.com
</p>

<p className="text-gray-400 mt-2">
GitHub: https://github.com/yourgithub
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