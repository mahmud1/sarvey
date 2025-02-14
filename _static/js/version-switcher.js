document.addEventListener("DOMContentLoaded", function () {
    const versionsJsonUrl = "https://https://mahmud1.github.io/sarvey/tags.json";

    // Create the dropdown container
    const dropdownContainer = document.createElement("div");
    dropdownContainer.id = "version-dropdown-container";
    dropdownContainer.style.position = "fixed";
    dropdownContainer.style.top = "10px";
    dropdownContainer.style.right = "10px";
    dropdownContainer.style.background = "#fff";
    dropdownContainer.style.padding = "5px";
    dropdownContainer.style.border = "1px solid #ccc";
    dropdownContainer.style.zIndex = "9999";

    // Create the dropdown
    const versionDropdown = document.createElement("select");
    versionDropdown.id = "version-selector";

    // Add default option
    const defaultOption = document.createElement("option");
    defaultOption.textContent =  Version";
    defaultOption.disabled = true;
    versionDropdown.appendChild(defaultOption);

    dropdownContainer.appendChild(versionDropdown);
    document.body.appendChild(dropdownContainer); // Add dropdown to the page

    // Fetch version data
    fetch(versionsJsonUrl)
        .then(response => response.json())
        .then(versions => {
            Object.keys(versions.tags).forEach(version => {
                let option = document.createElement("option");
                option.value = versions.tags[version];
                option.textContent = version;
                versionDropdown.appendChild(option);
            });

            // Load stored version selection
            const storedVersion = localStorage.getItem("selectedVersion");
            if (storedVersion) {
                versionDropdown.value = storedVersion;
            }

            versionDropdown.addEventListener("change", function () {
                let selectedUrl = versionDropdown.value;
                localStorage.setItem("selectedVersion", selectedUrl);
                window.location.href = selectedUrl; // Redirect to the new version
            });
        })
        .catch(error => console.error("Error loading versions:", error));
});
