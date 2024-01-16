<script>
    const audioFile = '';

    function handleFileChange(event) {
        audioFile = event.target.files[0];
        console.log('File selected:', audioFile.name);
        // Optional: Code to send this file to the server
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
