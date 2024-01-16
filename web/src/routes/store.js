import { writable } from 'svelte/store';
import { browser } from '$app/environment';


    export const token = writable(browser && localStorage.getItem("userId") || "webjeda");
    export const darkMode = writable(browser && localStorage.getItem('darkMode') === 'true' || false);
    
// content.subscribe((value) => localStorage.content = value)
// export const loggedUser = writable(JSON.parse(safeLocalStorage('loggedUser') || '{}') || {});
// export const userType = writable(safeLocalStorage('userType') || '');
// export const darkMode = writable(safeLocalStorage('darkMode') === 'true' || false);

// if (typeof window !== 'undefined') {
//     token.subscribe((token) => localStorage.setItem('token', token));
//     loggedUser.subscribe((user) => localStorage.setItem('loggedUser', JSON.stringify(user)));
//     userType.subscribe((userType) => localStorage.setItem('userType', userType));
//     darkMode.subscribe((darkMode) => localStorage.setItem('darkMode', darkMode));
// }
darkMode.subscribe((darkMode) => {
    if (browser){
        localStorage.setItem('darkMode', darkMode)
    }
});

// export const serverError = writable('');
// export const toast = writable({ show: false, message: '', type: '' });

token.subscribe((token) => {
    if (browser){
        localStorage.setItem('token', token);
    }
});

export const serverError = writable('');
export const toast = writable({ show: false, message: '', type: '' });