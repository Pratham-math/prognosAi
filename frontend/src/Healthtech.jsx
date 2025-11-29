// App.js
import React from 'react';

const HealthTechLanding = () => {
  const hospitals = [
    {
      id: 1,
      name: "City General Hospital",
      address: "123 Healthcare Ave, Medical District",
      phone: "+1 (555) 123-4567",
      email: "info@citygeneral.com",
      image: "https://images.unsplash.com/photo-1519494026892-80bbd2d6fd0d?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80"
    },
    // ... rest of the hospitals array
  ];

  const features = [
    // ... features array remains the same
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100 text-gray-800">
      {/* Header */}
      <header className="bg-white shadow-lg sticky top-0 z-50">
        <div className="container mx-auto px-4 py-4">
          <div className="flex justify-between items-center">
            <div className="flex items-center space-x-2">
              <div className="w-10 h-10 bg-blue-600 rounded-full flex items-center justify-center">
                <span className="text-white font-bold text-lg">H</span>
              </div>
              <span className="text-2xl font-bold text-gray-900">HealthAI</span>
            </div>
            <nav className="hidden md:flex space-x-8">
              <a href="#solutions" className="text-gray-600 hover:text-blue-600 font-medium">Solutions</a>
              <a href="#hospitals" className="text-gray-600 hover:text-blue-600 font-medium">Hospitals</a>
              <a href="#contact" className="text-gray-600 hover:text-blue-600 font-medium">Contact</a>
            </nav>
            <button className="bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700 transition duration-300">
              Get Started
            </button>
          </div>
        </div>
      </header>

      {/* Hero Section */}
      <section className="py-20 px-4">
        <div className="container mx-auto text-center">
          <h1 className="text-5xl md:text-6xl font-bold text-gray-900 mb-6">
            Revolutionizing Healthcare with
            <span className="text-blue-600 block">AI-Powered Solutions</span>
          </h1>
          <p className="text-xl text-gray-600 mb-8 max-w-3xl mx-auto">
            Manage unpredictable patient surges during festivals, pollution spikes, or epidemics with our intelligent AI agent that anticipates healthcare demands.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <button className="bg-blue-600 text-white px-8 py-3 rounded-lg hover:bg-blue-700 transition duration-300 text-lg font-semibold">
              Schedule Demo
            </button>
            <button className="border-2 border-blue-600 text-blue-600 px-8 py-3 rounded-lg hover:bg-blue-50 transition duration-300 text-lg font-semibold">
              Learn More
            </button>
          </div>
        </div>
      </section>

      {/* Problem Statement Section */}
      <section id="solutions" className="py-16 bg-white border-t border-gray-200">
        <div className="container mx-auto px-4">
          <div className="text-center mb-12">
            <h2 className="text-4xl font-bold text-gray-900 mb-4">Solving Critical Healthcare Challenges</h2>
            <p className="text-xl text-gray-600 max-w-3xl mx-auto">
              Our AI agent autonomously analyzes data to recommend staffing, supply, and patient advisory actions in advance.
            </p>
          </div>

          <div className="grid md:grid-cols-2 gap-12 items-center">
            <div>
              <img 
                src="https://images.unsplash.com/photo-1576091160399-112ba8d25d1f?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80"
                alt="Healthcare AI Dashboard"
                className="rounded-2xl shadow-2xl"
              />
            </div>
            <div className="space-y-6">
              <div className="bg-gray-50 p-6 rounded-2xl border border-gray-200 shadow-sm">
                <h3 className="text-2xl font-semibold text-blue-600 mb-3">Problem Statement 1</h3>
                <p className="text-gray-700">
                  Manage unpredictable surges in patients during festivals, pollution spikes, or epidemics with an AI agent that autonomously analyzes data and recommends staffing, supply, and patient advisory actions in advance.
                </p>
              </div>
              <div className="bg-gray-50 p-6 rounded-2xl border border-gray-200 shadow-sm">
                <h3 className="text-2xl font-semibold text-green-600 mb-3">Problem Statement 2</h3>
                <p className="text-gray-700">
                  Bring your own problem in Healthtech, leveraging Agentic AI to create customized solutions for your specific healthcare challenges.
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-16 bg-gradient-to-r from-gray-50 to-white">
        <div className="container mx-auto px-4">
          <div className="text-center text-gray-900 mb-12">
            <h2 className="text-4xl font-bold mb-4">How Our AI Solution Works</h2>
            <p className="text-xl text-gray-600">Intelligent features that transform healthcare management</p>
          </div>
          <div className="grid md:grid-cols-3 gap-8">
            {features.map((feature, index) => (
              <div key={index} className="bg-white rounded-2xl p-8 shadow-lg transform hover:scale-105 transition duration-300 border border-gray-200">
                <div className="text-4xl mb-4">{feature.icon}</div>
                <h3 className="text-2xl font-bold text-gray-900 mb-3">{feature.title}</h3>
                <p className="text-gray-600">{feature.description}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Hospital Contacts Section */}
      <section id="hospitals" className="py-16 bg-gray-50">
        <div className="container mx-auto px-4">
          <div className="text-center mb-12">
            <h2 className="text-4xl font-bold text-gray-900 mb-4">Partner Hospitals</h2>
            <p className="text-xl text-gray-600">Leading healthcare institutions using our AI solutions</p>
          </div>
          <div className="grid md:grid-cols-3 gap-8">
            {hospitals.map((hospital) => (
              <div key={hospital.id} className="bg-white rounded-2xl overflow-hidden shadow-lg hover:shadow-xl transition duration-300 border border-gray-200">
                <img 
                  src={hospital.image} 
                  alt={hospital.name}
                  className="w-full h-48 object-cover"
                />
                <div className="p-6">
                  <h3 className="text-2xl font-bold text-gray-900 mb-3">{hospital.name}</h3>
                  <div className="space-y-2 text-gray-600">
                    <p className="flex items-center">
                      <span className="mr-2">📍</span>
                      {hospital.address}
                    </p>
                    <p className="flex items-center">
                      <span className="mr-2">📞</span>
                      {hospital.phone}
                    </p>
                    <p className="flex items-center">
                      <span className="mr-2">✉️</span>
                      {hospital.email}
                    </p>
                  </div>
                  <button className="w-full mt-4 bg-blue-600 text-white py-2 rounded-lg hover:bg-blue-700 transition duration-300">
                    Contact Hospital
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 bg-gradient-to-r from-gray-50 to-white border-t border-gray-200">
        <div className="container mx-auto px-4 text-center">
          <h2 className="text-4xl md:text-5xl font-bold text-gray-900 mb-6">
            Ready to Transform Your Healthcare Services?
          </h2>
          <p className="text-xl text-gray-600 mb-8 max-w-2xl mx-auto">
            Join leading hospitals already using our AI-powered platform to manage patient surges and optimize resources.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <button className="bg-blue-600 text-white px-8 py-3 rounded-lg hover:bg-blue-700 transition duration-300 text-lg font-semibold">
              Request Demo
            </button>
            <button className="border-2 border-blue-600 text-blue-600 px-8 py-3 rounded-lg hover:bg-blue-50 transition duration-300 text-lg font-semibold">
              Contact Sales
            </button>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer id="contact" className="bg-gray-100 text-gray-800 py-12 border-t border-gray-300">
        <div className="container mx-auto px-4">
          <div className="grid md:grid-cols-4 gap-8">
            <div>
              <div className="flex items-center space-x-2 mb-4">
                <div className="w-8 h-8 bg-blue-600 rounded-full flex items-center justify-center">
                  <span className="text-white font-bold">H</span>
                </div>
                <span className="text-xl font-bold text-gray-900">HealthAI</span>
              </div>
              <p className="text-gray-600">
                Transforming healthcare through intelligent AI solutions and predictive analytics.
              </p>
            </div>
            {/* Rest of the footer content remains the same */}
          </div>
          <div className="border-t border-gray-300 mt-8 pt-8 text-center text-gray-500">
            <p>&copy; 2024 HealthAI. All rights reserved.</p>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default HealthTechLanding;