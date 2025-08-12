document.addEventListener('DOMContentLoaded', function() {
    // Handle form submission
    const form = document.querySelector('form');
    if (form) {
        form.addEventListener('submit', function(e) {
            const patientIdInput = document.getElementById('patient_id');
            if (patientIdInput) {
                const patientId = parseInt(patientIdInput.value);
                if (isNaN(patientId)) {
                    e.preventDefault();
                    alert('Please enter a valid patient ID');
                }
            }
        });
    }
    
    // Add animations to elements
    const animateElements = document.querySelectorAll('.animate-in');
    animateElements.forEach((el, index) => {
        el.style.animationDelay = `${index * 0.1}s`;
    });
    
    // Smooth scrolling for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth'
                });
            }
        });
    });
});