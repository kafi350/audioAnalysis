import { token, serverError } from './store';
import { get } from 'svelte/store';

const BASE_URL = 'http://localhost:8000/api';

const getHeader = () => {
	return {
		headers: {
			Authorization: 'Bearer ' + get(token),
			'Content-Type': 'application/json'
		}
	};
};

/**
 * Handles errors that occur during API requests.
 * If the error status is 401 or 403, it clears the token and logged user data.
 * If the error can be parsed as JSON, it throws the parsed error.
 * Otherwise, it sets the server error and throws the original error.
 *
 * @param {Error} err - The error object.
 * @throws {Error} - The original error or the parsed error.
 */
const errorHandler = (err) => {
	if (err.status === 401 || err.status === 403) {
		token.set('');
		loggedUser.set({});
	}
	// try to get the error as json
	try {
		return err.json().then((error) => {
			throw error;
		});
	} catch (e) {
		serverError.set(err);
		throw err;
	}
};

const fetchWithHeader = (url, options) => {
	return fetch(url, { ...options, ...getHeader() })
		.then((response) => {
			if (!response.ok) {
				throw response;
			}
			serverError.set('');
			return response.json();
		})
		.catch(errorHandler);
};

const AudioAnalysis ={
	upload: (formdata) => 
		fetch(`${BASE_URL}/audio/upload`, {
			method: 'POST',
			body: formdata,
			headers: {
                Authorization: 'Bearer ' + get(token),
            },
		}),
	genderDetection : (formData) =>
		fetch(`${BASE_URL}/audio/genderdetection`, {
			method: 'POST',
			body: formData,
			headers: {
				Authorization: 'Bearer ' + get(token),
			},
		})
		.then((response) => {
			if (!response.ok) {
				throw response;
			}
			serverError.set('');
			return response.json();
		})
		.catch(errorHandler),		
	classify: (formData) =>
		fetch(`${BASE_URL}/audio/classify`, {
			method: 'POST',
			body: formData,
			headers: {
				Authorization: 'Bearer ' + get(token),
			},
		})
		.then((response) => {
			if (!response.ok) {
				throw response;
			}
			serverError.set('');
			return response.json();
		})
		.catch(errorHandler),
	emotionDetection: (formData) =>
		fetch(`${BASE_URL}/audio/emotiondetection`, {
			method: 'POST',
			body: formData,
			headers: {
				Authorization: 'Bearer ' + get(token),
			},
		})
		.then((response) => {
			if (!response.ok) {
				throw response;
			}
			serverError.set('');
			return response.json();
		})
		.catch(errorHandler),
};

export default {
    AudioAnalysis
};