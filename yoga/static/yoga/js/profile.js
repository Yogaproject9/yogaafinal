let profileimagebutton = document.querySelector(".profileheadericon");
let popupcard = document.querySelector(".profilecard");

profileimagebutton.addEventListener('click',()=> {
    popupcard.classList.toggle("profilecardupdated");
})