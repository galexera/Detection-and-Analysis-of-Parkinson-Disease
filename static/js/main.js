

$(document).ready(function () {
    // Init
    $('.image-section').hide();
    $('.loader').hide();
    $('#result').hide();

    // Upload Preview
    function readURL(input) {
        if (input.files && input.files[0]) {
            var reader = new FileReader();
            reader.onload = function (e) {
                $('#imagePreview').css('background-image', 'url(' + e.target.result + ')');
                $('#imagePreview').hide();
                $('#imagePreview').fadeIn(650);
            }
            reader.readAsDataURL(input.files[0]);
        }
    }
    
    $("#imageUpload").change(function () {
        $('.image-section').show();
        $('#btn-predict-spiral_cfs').show();
        $('#result').text('');
        $('#result').hide();
        readURL(this);
    });

    $("#imageUpload").change(function () {
        $('.image-section').show();
        $('#btn-predict-spiral').show();
        $('#result').text('');
        $('#result').hide();
        readURL(this);
    });

    $("#imageUpload").change(function () {
        $('.image-section').show();
        $('#btn-predict-vocal').show();
        $('#result').text('');
        $('#result').hide();
        readURL(this);
    });

    $("#imageUpload").change(function () {
        $('.image-section').show();
        $('#btn-predict-vocal_cfs').show();
        $('#result').text('');
        $('#result').hide();
        readURL(this);
    });

    // single spiral detect
    $('#btn-predict-spiral').click(function () {
        var form_data = new FormData($('#upload-file')[0]);
        // Show loading animation
        $(this).hide();
        $('.loader').show();
        // Make prediction by calling api /predict
        $.ajax({
            type: 'POST',
            url: '/spiral',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            async: true,
            success: function (data) {
                // Get and display the result
                $('.loader').hide();
                $('#result').fadeIn(600);
                $('#result').text(' RESULT:  ' + data);
                console.log('Success!');
            },
        });
    });
    
    //spiral cfs
    $('#btn-predict-spiral_cfs').click(function () {
        var form_data = new FormData($('#upload-file')[0]);
        // Show loading animation
        $(this).hide();
        $('.loader').show();
        // Make prediction by calling api /predict
        $.ajax({
            type: 'POST',
            url: '/cfs_spiral_predict',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            async: true,
            success: function (data) {
                // Get and display the result
                $('.loader').hide();
                $('#result').fadeIn(600);
                // $('#result').text(' RESULT:  ' + data);
                console.log('Success!');
            },
        });
    });

    //single vocal detect
    $('#btn-predict-vocal').click(function () {
        var form_data = new FormData($('#upload-file')[0]);
        // Show loading animation
        $(this).hide();
        $('.loader').show();
        // Make prediction by calling api /predict
        $.ajax({
            type: 'POST',
            url: '/vocal',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            async: true,
            success: function (data) {
                // Get and display the result
                $('.loader').hide();
                $('#result').fadeIn(600);
                $('#result').text(' RESULT:  ' + data);
                console.log('Success!');
            },
        });
    });
    
    //cfs vocal

    $('#btn-predict-vocal_cfs').click(function () {
        var form_data = new FormData($('#upload-file')[0]);
        // Show loading animation
        $(this).hide();
        $('.loader').show();
        // Make prediction by calling api /predict
        $.ajax({
            type: 'POST',
            url: '/cfs_vocal_predict',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            async: true,
            success: function (data) {
                // Get and display the result
                $('.loader').hide();
                $('#result').fadeIn(600);
                // $('#result').text(' RESULT:  ' + data);
                console.log('Success!');
            },
        });
    });

        //single Motor-movement detect
        $('#btn-predict_MM').click(function () {
            var form_data = new FormData($('#upload-file')[0]);
            // Show loading animation
            $(this).hide();
            $('.loader').show();
            // Make prediction by calling api /predict
            $.ajax({
                type: 'POST',
                url: '/MM',
                data: form_data,
                contentType: false,
                cache: false,
                processData: false,
                async: true,
                success: function (data) {
                    // Get and display the result
                    $('.loader').hide();
                    $('#result').fadeIn(600);
                    $('#result').text(' RESULT:  ' + data);
                    console.log('Success!');
                },
            });
        });
        // function launch_toast() {
        //     var x = document.getElementById("toast")
        //     x.className = "show";
        //     setTimeout(function(){ x.className = x.className.replace("show", ""); }, 5000);
        // }
        
        //cfs motor
        $('.close-btn').click(function(){
            $('.alert').removeClass("show");
            $('.alert').addClass("hide");
          });
        
        //back to home
        // $('#exit').click(function () {
        //     var form_data = new FormData($('#upload-file')[0]);
        //     // Show loading animation
        //     $(this).hide();
        //     $('.loader').show();
        //     // Make prediction by calling api /predict
        //     $.ajax({
        //         type: 'POST',
        //         url: '/MM',
        //         data: form_data,
        //         contentType: false,
        //         cache: false,
        //         processData: false,
        //         async: true,
        //         success: function (data) {
        //             // Get and display the result
        //             $('.loader').hide();
        //             $('#result').fadeIn(600);
        //             $('#result').text(' RESULT:  ' + data);
        //             console.log('Success!');
        //         },
        //     });
        // });
        $('#btn-predict_MM_cfs').click(function () {
            var form_data = new FormData($('#upload-file')[0]);
            // Show loading animation
            $(this).hide();
            $('.loader').show();

            $('.alert').addClass("show");
            console.log("Poppp")
            $('.alert').removeClass("hide");
            $('.alert').addClass("showAlert");
            setTimeout(function(){
              $('.alert').removeClass("show");
              $('.alert').addClass("hide");
            },5000);

            // Make prediction by calling api /predict
            // launch_toast();
            $.ajax({
                type: 'POST',
                url: '/cfs_MM_predict',
                data: form_data,
                contentType: false,
                cache: false,
                processData: false,
                async: true,
                success: function (data) {
                    // Get and display the result
                    $('.loader').hide();
                    $('#result').fadeIn(600);
                    // $('#result').text(' RESULT:  ' + data);
                    $('#result').show();

                    console.log('Success!');
                },
            });
        });

});
