<script>
    import axios from 'axios'; // Make sure to install axios using npm install axios
    import client from '../client.js';

    let audioFile = '';

    
    function handleFileChange(event) {
        audioFile = event.target.files[0];
        console.log('File selected:', audioFile.name);

            // Create a FormData object
        var formdata = new FormData();
        formdata.append("file", event.target.files[0], event.target.files[0].name);
        formdata.append("taken_at", new Date().toISOString())
        
        client.AudioAnalysis.upload(formdata)
            .then(response => {
                console.log(response);
            })
            .catch(error => {
                console.log(error);
            });
            
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
