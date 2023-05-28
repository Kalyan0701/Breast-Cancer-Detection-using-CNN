const dropdown = document.getElementById("logo-dropdown");

document.getElementById("logo-btn").addEventListener("click", () => {
    if(dropdown.style.display === 'block'){
        dropdown.style.display = "none";
    }else{
        dropdown.style.display = "block"
    }
})



console.log(dropdown)