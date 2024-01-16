<script>
    import axios from 'axios'; // Make sure to install axios using npm install axios

    let audioFile = '';

    async function handleFileChange(event) {
        audioFile = event.target.files[0];
        console.log('File selected:', audioFile.name);

        // Create a FormData object
        let formData = new FormData();

        // Append the file to the FormData object
        formData.append('file', audioFile);

        // Send the file to the server
        try {
            const response = await axios.post('/upload', formData, {
                headers: {
                    'Content-Type': 'multipart/form-data'
                }
            });
            console.log('File uploaded successfully:', response.data);
        } catch (error) {
            console.error('Error uploading file:', error);
        }
    }
</script>

<main>
    <h1>Upload Audio File</h1>
    <input type="file" accept="audio/*" on:change={handleFileChange} />
    {#if audioFile}
        <p>File selected: {audioFile.name}</p>
        <!-- Display more file details or options here -->
    {/if}
</main>

<style>
    main {
        text-align: center;
        padding: 1em;
        max-width: 400px;
        margin: auto;
    }
    input {
        margin-top: 1em;
    }
</style>
