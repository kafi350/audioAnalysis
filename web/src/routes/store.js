import { writable } from 'svelte/store';
import { browser } from '$app/environment';


export const token = writable(browser && localStorage.getItem("audioData") || "webjeda");
export const darkMode = writable(browser && localStorage.getItem('darkMode') === 'true' || false);
    
darkMode.subscribe((darkMode) => {
    if (browser){
        localStorage.setItem('darkMode', darkMode)
    }
});

token.subscribe((token) => {
    if (browser){
        localStorage.setItem('token', token);
    }
});

export const serverError = writable('');
export const toast = writable({ show: false, message: '', type: '' });
